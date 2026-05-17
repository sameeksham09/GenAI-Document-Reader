# app_ui.py
import streamlit as st
import pickle
import os

from retriever import retrieve_context, add_new_document, get_all_documents, delete_document
from prompts import get_instruction, build_prompt, build_rewrite_prompt
from llm_utils import generate_answer, generate_answer_stream
from logger import log_query, load_logs, get_stats

DOC_META_FILE = "doc_metadata.pkl"

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄 RAG QA System")
st.markdown("Ask questions based on your documents and get grounded answers with citations.")

# ── Session State ──────────────────────────────────────────────────────────────
if "selected_doc"         not in st.session_state: st.session_state.selected_doc         = None
if "last_context"         not in st.session_state: st.session_state.last_context         = None
if "last_question"        not in st.session_state: st.session_state.last_question        = None
if "chat_history"         not in st.session_state: st.session_state.chat_history         = []
if "last_rewritten_query" not in st.session_state: st.session_state.last_rewritten_query = None
if "confirm_delete"       not in st.session_state: st.session_state.confirm_delete       = False

# ── Load metadata ──────────────────────────────────────────────────────────────
def load_metadata():
    if os.path.exists(DOC_META_FILE):
        with open(DOC_META_FILE, "rb") as f:
            return pickle.load(f)
    return {}

metadata = load_metadata()

# ── Layout ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# ══════════════════════════════════════════════════════
# LEFT COLUMN
# ══════════════════════════════════════════════════════
with col1:

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("Upload a new document (TXT or PDF)")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

    if uploaded_file:
        with st.spinner("Processing and adding document..."):
            added = add_new_document(uploaded_file.read(), uploaded_file.name)

        if added:
            st.success(f"✅ {uploaded_file.name} added successfully!")
            st.session_state.selected_doc = uploaded_file.name
            metadata = load_metadata()
        else:
            st.session_state.selected_doc = uploaded_file.name
            doc_list = get_all_documents()
            if uploaded_file.name in doc_list:
                st.info(f"📋 **{uploaded_file.name}** is already indexed and ready to use.")

    # ── Document selector ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # WHAT CHANGED — doc list source
    #
    # BEFORE: loaded uploaded_docs.pkl (a separate pickle file we maintained)
    # NOW:    calls get_all_documents() which queries ChromaDB metadata directly
    #         Single source of truth — no pkl file to get out of sync
    # ─────────────────────────────────────────────────────────────────────────
    doc_list = get_all_documents()

    if doc_list:
        st.divider()
        st.subheader("📂 Indexed Documents")

        chosen = st.selectbox(
            "Select active document:",
            options=doc_list,
            index=doc_list.index(st.session_state.selected_doc)
                  if st.session_state.selected_doc in doc_list else 0
        )
        if chosen != st.session_state.selected_doc:
            st.session_state.selected_doc  = chosen
            st.session_state.chat_history  = []
            st.session_state.last_context  = None
            st.session_state.last_question = None
            st.session_state.confirm_delete = False

        # ── Document deletion ─────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # NEW FEATURE — Document Deletion
        #
        # BEFORE (FAISS): Impossible without wiping the entire index.
        #   When the resume got mixed in, the only fix was:
        #     rm -f chunks.pkl doc_index.faiss uploaded_docs.pkl doc_metadata.pkl
        #   That deleted EVERYTHING including the DBMS document.
        #
        # NOW (ChromaDB): Surgical delete in one line.
        #   collection.delete(where={"source": filename})
        #   Only that document's chunks are removed. Everything else untouched.
        #
        # UI pattern: two-step confirmation (click Delete → confirm)
        #   Prevents accidental deletion. Standard UX pattern for destructive actions.
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🗑️ Document Management")

        if not st.session_state.confirm_delete:
            if st.button(f"🗑️ Delete '{chosen}' from index"):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning(f"Are you sure you want to delete **{chosen}**? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, delete it"):
                    delete_document(chosen)
                    st.session_state.selected_doc   = None
                    st.session_state.chat_history   = []
                    st.session_state.last_context   = None
                    st.session_state.last_question  = None
                    st.session_state.confirm_delete = False
                    st.success(f"✅ '{chosen}' deleted from index.")
                    st.rerun()
            with c2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_delete = False
                    st.rerun()

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Document Summary")
    if not st.session_state.selected_doc:
        st.info("Upload a document to view its summary.")
    else:
        summary = metadata.get(st.session_state.selected_doc, {}).get("summary", "No summary available.")
        st.markdown(summary)

    # ── Clear conversation ────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.chat_history         = []
        st.session_state.last_context         = None
        st.session_state.last_question        = None
        st.session_state.last_rewritten_query = None
        st.success("Conversation cleared.")

    # ── Activity Log ──────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # NEW FEATURE — Activity Log
    #
    # Shows a real-time record of all uploads, deletes, and queries.
    # Loaded from activity_log.jsonl — one JSON line per event.
    #
    # Stats panel: total uploads, deletes, queries, most queried doc.
    # Event list : last 20 events in reverse chronological order (newest first).
    #
    # WHY IN AN EXPANDER:
    #   The left column is already busy. An expander keeps it clean —
    #   users who want the log can open it, others never see it.
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Activity Log", expanded=False):

        # ── Stats ─────────────────────────────────────────────────────────
        stats = get_stats()
        s1, s2, s3 = st.columns(3)
        s1.metric("📤 Uploads",  stats["total_uploads"])
        s2.metric("❓ Queries",  stats["total_queries"])
        s3.metric("🗑️ Deletes",  stats["total_deletes"])

        if stats["most_queried"] != "—":
            st.caption(f"Most queried: **{stats['most_queried']}**")

        st.divider()

        # ── Event list ────────────────────────────────────────────────────
        logs = load_logs(last_n=20)
        if not logs:
            st.info("No activity yet. Upload a document to get started.")
        else:
            # Show newest first
            for entry in reversed(logs):
                ts    = entry.get("timestamp", "")
                event = entry.get("event", "")

                if event == "upload":
                    st.markdown(
                        f"✅ `{ts}` **Uploaded** {entry.get('filename','')} "
                        f"— {entry.get('chunks', 0)} chunks"
                    )
                elif event == "delete":
                    st.markdown(
                        f"🗑️ `{ts}` **Deleted** {entry.get('filename','')}"
                    )
                elif event == "query":
                    q   = entry.get("question", "")[:60]
                    doc = entry.get("doc", "")
                    st.markdown(
                        f"❓ `{ts}` **Asked** \"{q}...\" → `{doc}`"
                    )

# ══════════════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════════════
with col2:
    st.subheader("Ask Questions")

    if not st.session_state.selected_doc:
        st.info("Please upload a document before asking questions.")
    else:
        active_doc = st.session_state.selected_doc
        st.caption(f"📄 Active document: **{active_doc}**")

        qtype = st.selectbox(
            "Select question type:",
            ["Descriptive", "MCQ", "True / False", "Fill in the blanks"]
        )
        qtype_map    = {"Descriptive": "1", "MCQ": "2", "True / False": "3", "Fill in the blanks": "4"}
        qtype_number = qtype_map[qtype]

        num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5) \
                        if qtype in ["MCQ", "True / False"] else 1

        question = st.text_input(
            "Enter your question:",
            placeholder="Ask something from this document..."
        )

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                instruction = get_instruction(qtype_number, num_questions)

                # ── Query rewriting ───────────────────────────────────────────
                if st.session_state.chat_history:
                    with st.spinner("🔄 Resolving question context..."):
                        rewrite_prompt  = build_rewrite_prompt(question, st.session_state.chat_history)
                        rewritten_query = generate_answer(rewrite_prompt, max_new_tokens=80)
                        lines = [l.strip() for l in rewritten_query.splitlines() if l.strip()]
                        rewritten_query = lines[0] if lines else question
                        for prefix in ["Rewritten query:", "Rewritten:", "Query:", "Question:"]:
                            if rewritten_query.lower().startswith(prefix.lower()):
                                rewritten_query = rewritten_query[len(prefix):].strip()
                        rewritten_query = rewritten_query.strip('"\'')
                        if len(rewritten_query) < 5:
                            rewritten_query = question
                    st.session_state.last_rewritten_query = rewritten_query
                else:
                    rewritten_query = question
                    st.session_state.last_rewritten_query = None

                # ── Retrieval ─────────────────────────────────────────────────
                with st.spinner("🔍 Searching document..."):
                    retrieved = retrieve_context(rewritten_query, selected_doc=active_doc)

                if not retrieved:
                    st.warning("I don't know based on the provided document.")
                else:
                    context = "\n".join(c["text"] for c in retrieved)
                    prompt  = build_prompt(
                        context, instruction, question,
                        chat_history=st.session_state.chat_history
                    )

                    # ── Streaming generation ──────────────────────────────────
                    # ─────────────────────────────────────────────────────────
                    # WHAT CHANGED — answer generation + display
                    #
                    # BEFORE:
                    #   with st.spinner("🤖 Generating answer..."):
                    #       answer = generate_answer(prompt)   # blocks 5-15s
                    #   st.text_area("Response", value=answer, height=250)
                    #
                    #   The entire UI froze until the LLM finished generating.
                    #   User saw nothing for 5-15 seconds then everything at once.
                    #
                    # NOW:
                    #   st.write_stream(generate_answer_stream(prompt))
                    #
                    #   generate_answer_stream() is a Python generator that
                    #   yields one token at a time from Ollama's streaming API.
                    #   st.write_stream() calls next() on the generator repeatedly
                    #   and appends each token to the UI immediately.
                    #
                    #   Result: first token appears in ~0.5 seconds.
                    #   User sees the answer building word by word — like ChatGPT.
                    #
                    # WHY WE STILL NEED generate_answer() (non-streaming):
                    #   We need the COMPLETE answer string to:
                    #   1. Save to chat_history for conversation memory
                    #   2. Log to activity_log.jsonl
                    #   st.write_stream() returns the full collected string
                    #   after streaming completes — we capture that.
                    # ─────────────────────────────────────────────────────────
                    st.markdown("### 🤖 AI Answer")
                    answer = st.write_stream(
                        generate_answer_stream(prompt, max_new_tokens=300)
                    )
                    # st.write_stream returns the full collected string
                    # once streaming is complete — use it for history + logging

                    st.session_state.last_context  = context
                    st.session_state.last_question = question
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                    st.session_state.chat_history = st.session_state.chat_history[-5:]

                    # Log the query event
                    log_query(
                        question=question,
                        rewritten_query=rewritten_query,
                        selected_doc=active_doc,
                        num_chunks_retrieved=len(retrieved)
                    )

                    if st.session_state.last_rewritten_query:
                        with st.expander("🔍 Retrieval query used (after rewriting)"):
                            st.caption(st.session_state.last_rewritten_query)

                    if len(st.session_state.chat_history) > 1:
                        st.markdown("### 💬 Conversation History")
                        for turn in st.session_state.chat_history[:-1]:
                            st.markdown(f"**Q:** {turn['question']}")
                            st.markdown(f"**A:** {turn['answer']}")
                            st.divider()

                    # ── Sources ───────────────────────────────────────────────
                    # ─────────────────────────────────────────────────────────
                    # WHAT CHANGED — distance metric label
                    #
                    # BEFORE: "FAISS dist" (L2 distance, lower = better)
                    # NOW:    "Cosine dist" (cosine distance, lower = better)
                    #
                    # Cosine distance = 1 - cosine_similarity
                    # Range: 0 (identical) to 2 (opposite)
                    # A score of 0.3 means 70% cosine similarity — good match.
                    # ─────────────────────────────────────────────────────────
                    st.markdown("### 📌 Sources")
                    for i, c in enumerate(retrieved, 1):
                        rrf_score    = c.get("score", 0)
                        rerank_score = c.get("rerank_score", None)
                        if rerank_score is not None:
                            st.write(
                                f"**#{i}** `{c['source']}` | "
                                f"RRF score: `{rrf_score:.4f}` | "
                                f"Rerank score: `{rerank_score:.4f}`"
                            )
                        else:
                            st.write(f"**#{i}** `{c['source']}` | RRF score: `{rrf_score:.4f}`")

    # ── Smart Follow-ups ───────────────────────────────────────────────────────
    if st.session_state.last_context and st.session_state.last_question:
        st.subheader("🔁 Smart Follow-ups")
        st.caption("Would you like:")
        f1, f2, f3 = st.columns(3)

        with f1:
            if st.button("1️⃣ Examples"):
                prompt = build_prompt(
                    st.session_state.last_context,
                    "Give clear, real-world examples based on the context.",
                    st.session_state.last_question,
                    chat_history=st.session_state.chat_history
                )
                st.text_area("📘 Examples", value=generate_answer(prompt), height=250)

        with f2:
            if st.button("2️⃣ MCQs"):
                prompt = build_prompt(
                    st.session_state.last_context,
                    get_instruction("2", 5),
                    st.session_state.last_question,
                    chat_history=st.session_state.chat_history
                )
                st.text_area("📝 MCQs", value=generate_answer(prompt), height=250)

        with f3:
            if st.button("3️⃣ Explain like I'm 5"):
                prompt = build_prompt(
                    st.session_state.last_context,
                    "Explain this in very simple terms like explaining to a 5-year-old.",
                    st.session_state.last_question,
                    chat_history=st.session_state.chat_history
                )
                st.text_area("🧸 ELI5", value=generate_answer(prompt), height=250)