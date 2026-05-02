# app_ui.py
import streamlit as st
import pickle
import os

from retriever import retrieve_context, add_new_document
from prompts import get_instruction, build_prompt, build_rewrite_prompt
from llm_utils import generate_answer

DOC_META_FILE = "doc_metadata.pkl"
DOC_LIST_FILE = "uploaded_docs.pkl"

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄 RAG QA System")
st.markdown("Ask questions based on your documents and get grounded answers with citations.")

# ── Session State ──────────────────────────────────────────────────────────────
if "selected_doc"         not in st.session_state: st.session_state.selected_doc         = None
if "last_context"         not in st.session_state: st.session_state.last_context         = None
if "last_question"        not in st.session_state: st.session_state.last_question        = None
if "chat_history"         not in st.session_state: st.session_state.chat_history         = []
if "last_rewritten_query" not in st.session_state: st.session_state.last_rewritten_query = None

# ── Load metadata ──────────────────────────────────────────────────────────────
def load_metadata():
    if os.path.exists(DOC_META_FILE):
        with open(DOC_META_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def load_doc_list():
    if os.path.exists(DOC_LIST_FILE):
        with open(DOC_LIST_FILE, "rb") as f:
            return pickle.load(f)
    return []

metadata = load_metadata()

# ── Layout ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# ══════════════════════════════════════════════════════
# LEFT COLUMN — Upload + doc selector + summary
# ══════════════════════════════════════════════════════
with col1:
    st.subheader("Upload a new document (TXT or PDF)")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

    if uploaded_file:
        with st.spinner("Processing and adding document..."):
            added = add_new_document(uploaded_file.read(), uploaded_file.name)

        if added:
            st.success(f"✅ {uploaded_file.name} added successfully!")
            st.session_state.selected_doc = uploaded_file.name
            metadata = load_metadata()   # reload after new upload
        else:
            # ─────────────────────────────────────────────────────────────────
            # KEY FIX — persistent selected_doc across Streamlit reruns
            #
            # PROBLEM:
            #   Streamlit reruns the entire script on every interaction
            #   (button click, text input, selectbox change, etc.).
            #   On rerun, file_uploader still holds the file, so the
            #   `if uploaded_file:` block executes again. But now
            #   add_new_document() returns False because the doc is already
            #   indexed (deduplication skips it). So `selected_doc` never
            #   gets set → right column shows "Please upload a document".
            #
            # FIX:
            #   When add_new_document returns False (already indexed),
            #   we still set selected_doc to the uploaded filename so the
            #   right column stays unlocked. The doc is already in FAISS —
            #   we just need to remember which one is active.
            # ─────────────────────────────────────────────────────────────────
            st.session_state.selected_doc = uploaded_file.name

            # Tell the user why there is no "added successfully" message.
            # It is not an error — the doc is already indexed and ready.
            doc_list_check = load_doc_list()
            if uploaded_file.name in doc_list_check:
                st.info(f"📋 **{uploaded_file.name}** is already indexed and ready to use.")

    # ── Document selector (if multiple docs uploaded) ─────────────────────────
    # Shows a dropdown of all previously indexed documents so the user can
    # switch between them without re-uploading.
    doc_list = load_doc_list()
    if doc_list:
        st.divider()
        st.subheader("📂 Select Active Document")
        chosen = st.selectbox(
            "Choose a document to query:",
            options=doc_list,
            index=doc_list.index(st.session_state.selected_doc)
                  if st.session_state.selected_doc in doc_list else 0
        )
        if chosen != st.session_state.selected_doc:
            st.session_state.selected_doc  = chosen
            st.session_state.chat_history  = []   # clear history when switching docs
            st.session_state.last_context  = None
            st.session_state.last_question = None

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Document Summary / Insights")
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

# ══════════════════════════════════════════════════════
# RIGHT COLUMN — Ask questions + output
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

                        # ── Extract just the first line ───────────────────────
                        # Small LLMs (TinyLlama etc.) often continue generating
                        # beyond the rewritten query — adding "Answer: ..." or
                        # repeating the conversation. We only want the very first
                        # non-empty line, which is always the rewritten query.
                        lines = [l.strip() for l in rewritten_query.splitlines() if l.strip()]
                        rewritten_query = lines[0] if lines else question

                        # Strip known label prefixes the LLM might echo
                        for prefix in ["Rewritten query:", "Rewritten:", "Query:", "Question:"]:
                            if rewritten_query.lower().startswith(prefix.lower()):
                                rewritten_query = rewritten_query[len(prefix):].strip()

                        # Strip surrounding quotes if present ("query" → query)
                        rewritten_query = rewritten_query.strip('"\'')

                        # Fallback: if result is still too short, use original
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

                    # ── Prompt + generation ───────────────────────────────────
                    prompt = build_prompt(
                        context, instruction, question,
                        chat_history=st.session_state.chat_history
                    )
                    with st.spinner("🤖 Generating answer..."):
                        answer = generate_answer(prompt)

                    # ── Save state ────────────────────────────────────────────
                    st.session_state.last_context  = context
                    st.session_state.last_question = question
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                    st.session_state.chat_history = st.session_state.chat_history[-5:]

                    # ── Rewritten query expander ──────────────────────────────
                    if st.session_state.last_rewritten_query:
                        with st.expander("🔍 Retrieval query used (after rewriting)"):
                            st.caption(st.session_state.last_rewritten_query)

                    # ── Conversation history ──────────────────────────────────
                    if len(st.session_state.chat_history) > 1:
                        st.markdown("### 💬 Conversation History")
                        for turn in st.session_state.chat_history[:-1]:
                            st.markdown(f"**Q:** {turn['question']}")
                            st.markdown(f"**A:** {turn['answer']}")
                            st.divider()

                    # ── Answer ────────────────────────────────────────────────
                    st.markdown("### 🤖 AI Answer")
                    st.text_area("Response", value=answer, height=250)

                    # ── Sources ───────────────────────────────────────────────
                    st.markdown("### 📌 Sources")
                    for i, c in enumerate(retrieved, 1):
                        faiss_score  = c.get("score", 0)
                        rerank_score = c.get("rerank_score", None)
                        if rerank_score is not None:
                            st.write(
                                f"**#{i}** `{c['source']}` | "
                                f"FAISS dist: `{faiss_score:.4f}` | "
                                f"Rerank score: `{rerank_score:.4f}`"
                            )
                        else:
                            st.write(f"**#{i}** `{c['source']}` | FAISS dist: `{faiss_score:.4f}`")

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