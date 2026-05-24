# app_ui.py
import streamlit as st
import pickle
import os
import time

from retriever import retrieve_context, add_new_document, get_all_documents, delete_document
from prompts import get_instruction, build_prompt, build_rewrite_prompt
from llm_utils import generate_answer, generate_answer_stream, check_faithfulness
from logger import log_query, load_logs, get_stats

DOC_META_FILE = "doc_metadata.pkl"

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem !important; max-width: 860px !important; margin: 0 auto !important; }
  .stApp { background: #f9f9f8; }

  [data-testid="stSidebar"] { background: #1f1e1d !important; border-right: none !important; }
  [data-testid="stSidebar"] * { color: #d4d2ce !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #ffffff !important; }

  [data-testid="stFileUploaderDropzone"] {
    background: #2d2c2b !important; border: 1.5px dashed #4a4845 !important; border-radius: 8px !important;
  }
  [data-testid="stFileUploaderDropzoneInstructions"] p,
  [data-testid="stFileUploaderDropzoneInstructions"] span { color: #9b9894 !important; font-size: 12px !important; }
  [data-testid="stFileUploader"] button {
    background: #2d2c2b !important; border: 1px solid #4a4845 !important;
    color: #d4d2ce !important; border-radius: 6px !important; font-size: 12px !important;
  }
  [data-testid="stFileUploader"] button:hover { background: #383634 !important; border-color: #6b6966 !important; }

  [data-testid="stSidebar"] .stButton button {
    background: #2d2c2b !important; border: 1px solid #3d3c3a !important; color: #c8c5c0 !important;
    border-radius: 6px !important; font-size: 12px !important; font-weight: 400 !important;
    transition: all 0.15s !important; text-align: left !important;
  }
  [data-testid="stSidebar"] .stButton button:hover { background: #383634 !important; color: #ffffff !important; border-color: #5a5856 !important; }

  .side-label { font-size: 10px !important; font-weight: 600 !important; color: #6b6966 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; margin: 0 0 8px 2px !important; }
  .side-logo { display: flex; align-items: center; gap: 10px; padding: 2px 0 14px; }
  .side-logo-box { width: 30px; height: 30px; background: #cc785c; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 16px; }
  .side-logo-name { font-size: 15px; font-weight: 500; color: #f0ede8 !important; }
  .side-logo-sub  { font-size: 11px; color: #6b6966 !important; }
  .side-doc-active { background: #2d2c2b; border: 1px solid #4a4845; border-radius: 7px; padding: 8px 10px; margin-bottom: 4px; }
  .side-doc-inactive { background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 8px 10px; margin-bottom: 4px; }
  .side-doc-inactive:hover { background: #272624; border-color: #3d3c3a; }
  .side-doc-name-active   { font-size: 12px; font-weight: 500; color: #f0ede8 !important; line-height: 1.4; }
  .side-doc-name-inactive { font-size: 12px; color: #9b9894 !important; line-height: 1.4; }
  .side-doc-meta { font-size: 11px; color: #6b6966 !important; margin-top: 2px; }
  .side-divider { border: none; border-top: 1px solid #2d2c2b; margin: 10px 0; }
  .side-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 4px 0 10px; }
  .side-stat  { background: #2d2c2b; border-radius: 7px; padding: 8px 6px; text-align: center; }
  .side-stat-n { font-size: 17px; font-weight: 500; color: #f0ede8 !important; }
  .side-stat-l { font-size: 10px; color: #6b6966 !important; text-transform: uppercase; letter-spacing: 0.05em; }

  .main-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh; text-align: center; }
  .main-empty-icon  { font-size: 44px; opacity: 0.5; margin-bottom: 14px; }
  .main-empty-title { font-size: 24px; font-weight: 500; color: #1f1e1d; margin-bottom: 8px; letter-spacing: -0.02em; }
  .main-empty-sub   { font-size: 14px; color: #8c8a85; max-width: 360px; line-height: 1.6; }
  .main-badges      { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 18px; }
  .main-badge       { background: #fff; border: 1px solid #e5e3de; border-radius: 20px; padding: 4px 12px; font-size: 12px; color: #6b6966; }

  .main-topbar      { margin-bottom: 16px; padding-bottom: 14px; border-bottom: 1px solid #e5e3de; }
  .main-topbar-name { font-size: 15px; font-weight: 500; color: #1f1e1d; }
  .main-topbar-sub  { font-size: 12px; color: #8c8a85; margin-top: 2px; }

  .chat-lbl   { font-size: 11px; font-weight: 500; color: #8c8a85; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
  .chat-lbl-r { font-size: 11px; font-weight: 500; color: #8c8a85; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; text-align: right; }
  .bubble-user {
    background: #cc785c; color: #ffffff;
    border-radius: 14px 4px 14px 14px;
    padding: 10px 15px; font-size: 14px; line-height: 1.6;
    max-width: 78%; margin-left: auto;
  }
  .bubble-asst {
    background: #ffffff; border: 1px solid #e5e3de; color: #1f1e1d;
    border-radius: 4px 14px 14px 14px;
    padding: 10px 15px; font-size: 14px; line-height: 1.65;
    max-width: 84%; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }

  .pill-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; }
  .pill-f { background: #f0faf4; color: #2d6a4f; border: 1px solid #a8d5b8; border-radius: 20px; padding: 2px 9px; font-size: 11px; }
  .pill-p { background: #fefaec; color: #7d5a00; border: 1px solid #f5d78a; border-radius: 20px; padding: 2px 9px; font-size: 11px; }
  .pill-u { background: #fdf2f2; color: #9b2c2c; border: 1px solid #f5a8a8; border-radius: 20px; padding: 2px 9px; font-size: 11px; }
  .pill-n { background: #f5f4f2; color: #6b6966; border: 1px solid #e5e3de; border-radius: 20px; padding: 2px 9px; font-size: 11px; }

  .src-wrap { background: #fff; border: 1px solid #e5e3de; border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
  .src-hdr  { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
  .src-num  { background: #cc785c; color: #fff; width: 19px; height: 19px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; flex-shrink: 0; }
  .src-file { font-size: 11px; color: #8c8a85; flex: 1; }
  .src-scr  { font-size: 11px; color: #cc785c; font-weight: 500; }
  .src-text { font-size: 12px; color: #4a4845; line-height: 1.55; }

  .stTextInput input {
    background: #ffffff !important; border: 1px solid #e5e3de !important;
    border-radius: 10px !important; color: #1f1e1d !important;
    font-size: 14px !important; padding: 10px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
  }
  .stTextInput input:focus {
    border-color: #cc785c !important;
    box-shadow: 0 0 0 3px rgba(204,120,92,0.1) !important;
    background: #fff !important;
  }
  .stTextInput input::placeholder { color: #aaa9a5 !important; }

  .stButton button[kind="primary"],
  .stButton button[data-testid="baseButton-primary"] {
    background: #cc785c !important; border: none !important; color: #ffffff !important;
    border-radius: 10px !important; font-size: 13px !important; font-weight: 500 !important;
    transition: background 0.15s !important;
  }
  .stButton button[kind="primary"]:hover,
  .stButton button[data-testid="baseButton-primary"]:hover { background: #b8674e !important; }

  .stButton button[kind="secondary"] {
    background: #ffffff !important; border: 1px solid #e5e3de !important; color: #4a4845 !important;
    border-radius: 8px !important; font-size: 12px !important; transition: all 0.15s !important;
  }
  .stButton button[kind="secondary"]:hover { background: #f5f4f2 !important; border-color: #ccc !important; }

  .main-divider { border: none; border-top: 1px solid #e5e3de; margin: 14px 0; }
  .qa-label { font-size: 10px; font-weight: 600; color: #8c8a85; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }

  .stDownloadButton button {
    background: #ffffff !important; border: 1px solid #e5e3de !important; color: #4a4845 !important;
    border-radius: 8px !important; font-size: 12px !important; transition: all 0.15s !important;
  }
  .stDownloadButton button:hover { background: #f5f4f2 !important; border-color: #ccc !important; }

  .stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important; border: 1px solid #e5e3de !important;
    border-radius: 8px !important; font-size: 13px !important;
  }

  /* Thinking indicator */
  .thinking-bar {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; background: #fff;
    border: 1px solid #e5e3de; border-radius: 4px 14px 14px 14px;
    max-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .thinking-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #cc785c;
    animation: bounce 1.2s infinite ease-in-out;
  }
  .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-6px); opacity: 1; }
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "selected_doc": None, "last_context": None, "last_question": None,
    "chat_history": [], "last_rewritten_query": None,
    "confirm_delete": False, "latency_data": None,
    "pending_question": None,   # question waiting to be processed
    "is_processing": False,     # flag: pipeline is running
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_metadata():
    if os.path.exists(DOC_META_FILE):
        with open(DOC_META_FILE, "rb") as f:
            return pickle.load(f)
    return {}

metadata = load_metadata()
stats    = get_stats()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="side-logo">
      <div class="side-logo-box">🧠</div>
      <div><div class="side-logo-name">DocuMind</div><div class="side-logo-sub">RAG · Document Q&A</div></div>
    </div>
    <hr class="side-divider">
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF or TXT", type=["txt","pdf"], label_visibility="collapsed")
    if uploaded_file:
        with st.spinner("Indexing..."):
            added = add_new_document(uploaded_file.read(), uploaded_file.name)
        if added:
            st.success("✅ Indexed!")
            st.session_state.selected_doc = uploaded_file.name
            metadata = load_metadata()
        else:
            st.session_state.selected_doc = uploaded_file.name
            if uploaded_file.name in get_all_documents():
                st.info("Already indexed.")

    st.markdown('<hr class="side-divider">', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Documents</div>', unsafe_allow_html=True)
    doc_list = get_all_documents()

    if not doc_list:
        st.markdown('<div style="font-size:12px;color:#6b6966;padding:4px 2px;">No documents yet.</div>', unsafe_allow_html=True)
    else:
        from retriever import collection as _col
        for doc in doc_list:
            is_active   = doc == st.session_state.selected_doc
            res         = _col.get(where={"source": doc})
            chunk_count = len(res["ids"]) if res["ids"] else 0
            short_name  = doc[:26] + "..." if len(doc) > 26 else doc
            card = "side-doc-active" if is_active else "side-doc-inactive"
            ncls = "side-doc-name-active" if is_active else "side-doc-name-inactive"
            st.markdown(f"""
            <div class="{card}">
              <div class="{ncls}">{short_name}</div>
              <div class="side-doc-meta">🧩 {chunk_count} chunks</div>
            </div>""", unsafe_allow_html=True)
            if not is_active:
                if st.button("Select", key=f"sel_{doc}", use_container_width=True):
                    st.session_state.selected_doc  = doc
                    st.session_state.chat_history  = []
                    st.session_state.last_context  = None
                    st.session_state.last_question = None
                    st.session_state.latency_data  = None
                    st.session_state.pending_question = None
                    st.session_state.is_processing = False
                    st.rerun()

    st.markdown('<hr class="side-divider">', unsafe_allow_html=True)

    if st.session_state.selected_doc and st.session_state.selected_doc in (doc_list or []):
        if not st.session_state.confirm_delete:
            if st.button("🗑 Delete active document", use_container_width=True):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning(f"Delete **{st.session_state.selected_doc[:22]}**?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes", use_container_width=True):
                    delete_document(st.session_state.selected_doc)
                    st.session_state.selected_doc   = None
                    st.session_state.chat_history   = []
                    st.session_state.confirm_delete = False
                    st.rerun()
            with c2:
                if st.button("No", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.rerun()

    st.markdown('<hr class="side-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="side-stats">
      <div class="side-stat"><div class="side-stat-n">{stats['total_uploads']}</div><div class="side-stat-l">Docs</div></div>
      <div class="side-stat"><div class="side-stat-n">{stats['total_queries']}</div><div class="side-stat-l">Queries</div></div>
    </div>""", unsafe_allow_html=True)

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.chat_history         = []
        st.session_state.last_context         = None
        st.session_state.last_question        = None
        st.session_state.last_rewritten_query = None
        st.session_state.latency_data         = None
        st.session_state.pending_question     = None
        st.session_state.is_processing        = False
        st.rerun()

    with st.expander("📋 Activity log"):
        logs = load_logs(last_n=15)
        if not logs:
            st.caption("No activity yet.")
        else:
            for entry in reversed(logs):
                ts    = entry.get("timestamp","")[-8:]
                event = entry.get("event","")
                if event == "upload":
                    st.caption(f"✅ {ts} · {entry.get('filename','')[:22]}")
                elif event == "delete":
                    st.caption(f"🗑 {ts} · {entry.get('filename','')[:22]}")
                elif event == "query":
                    st.caption(f"❓ {ts} · {entry.get('question','')[:30]}...")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.selected_doc:
    st.markdown("""
    <div class="main-empty">
      <div class="main-empty-icon">🧠</div>
      <div class="main-empty-title">DocuMind</div>
      <div class="main-empty-sub">Upload a document from the sidebar, then ask anything about its contents.</div>
      <div class="main-badges">
        <span class="main-badge">🔍 Hybrid BM25 + Dense</span>
        <span class="main-badge">⚡ Cross-Encoder Reranking</span>
        <span class="main-badge">💬 Conversation Memory</span>
        <span class="main-badge">🛡 Faithfulness Check</span>
        <span class="main-badge">🤖 LLaMA 3.2</span>
      </div>
    </div>""", unsafe_allow_html=True)

else:
    active_doc = st.session_state.selected_doc

    # ── Topbar ────────────────────────────────────────────────────────────────
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.markdown(f"""
        <div class="main-topbar">
          <div class="main-topbar-name">📄 {active_doc}</div>
          <div class="main-topbar-sub">Ask anything about this document</div>
        </div>""", unsafe_allow_html=True)
    with tcol2:
        qtype = st.selectbox("Type", ["Descriptive","MCQ","True / False","Fill in the blanks"], label_visibility="collapsed")
        num_q = st.number_input("Count", 1, 20, 5, label_visibility="collapsed") if qtype in ["MCQ","True / False"] else 1

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Render all settled history (always at top)
    # ══════════════════════════════════════════════════════════════════════════
    if not st.session_state.chat_history and not st.session_state.is_processing:
        st.markdown("""
        <div style="padding:40px 0 24px;text-align:center;">
          <div style="font-size:32px;opacity:0.3;margin-bottom:10px;">🧠</div>
          <div style="font-size:15px;font-weight:500;color:#4a4845;margin-bottom:4px;">What would you like to know?</div>
          <div style="font-size:12px;color:#aaa9a5;">Ask anything about this document</div>
        </div>""", unsafe_allow_html=True)
    else:
        for turn in st.session_state.chat_history:
            sc = turn.get("faith",{}).get("score",5)
            if sc >= 7:   fp = f'<span class="pill-f">✓ Faithful {sc}/10</span>'
            elif sc >= 4: fp = f'<span class="pill-p">~ Partial {sc}/10</span>'
            else:         fp = f'<span class="pill-u">✗ Not faithful {sc}/10</span>'
            lat  = f'<span class="pill-n">⚡ {turn["latency"]}ms</span>' if turn.get("latency") else ""
            srcs = f'<span class="pill-n">📌 {turn["sources_count"]} sources</span>' if turn.get("sources_count") else ""
            st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:flex-end;margin:14px 0;">
              <div class="chat-lbl-r">You</div>
              <div class="bubble-user">{turn['question']}</div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-start;margin:14px 0;">
              <div class="chat-lbl">🧠 DocuMind</div>
              <div class="bubble-asst">{turn['answer']}</div>
              <div class="pill-row">{fp}{lat}{srcs}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — If processing: show pending question + thinking dots, run pipeline
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.is_processing and st.session_state.pending_question:
        pq = st.session_state.pending_question

        # Show the user bubble for the in-flight question
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:flex-end;margin:14px 0;">
          <div class="chat-lbl-r">You</div>
          <div class="bubble-user">{pq}</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-start;margin:14px 0;">
          <div class="chat-lbl">🧠 DocuMind</div>
          <div class="thinking-bar">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Run the full pipeline ──────────────────────────────────────────
        qtype_map   = {"Descriptive":"1","MCQ":"2","True / False":"3","Fill in the blanks":"4"}
        instruction = get_instruction(qtype_map[qtype], num_q)
        t_start     = time.time()

        # Query rewriting
        if st.session_state.chat_history:
            rp = build_rewrite_prompt(pq, st.session_state.chat_history)
            rq = generate_answer(rp, max_new_tokens=80)
            lines = [l.strip() for l in rq.splitlines() if l.strip()]
            rq = lines[0] if lines else pq
            for pfx in ["Rewritten query:","Rewritten:","Query:","Question:"]:
                if rq.lower().startswith(pfx.lower()):
                    rq = rq[len(pfx):].strip()
            rq = rq.strip('"\'')
            if len(rq) < 5: rq = pq
        else:
            rq = pq

        # Retrieval
        t_r0 = time.time()
        retrieved = retrieve_context(rq, selected_doc=active_doc)
        t_retrieve = round((time.time() - t_r0) * 1000)

        if not retrieved:
            # No results — save empty turn and rerun
            st.session_state.is_processing    = False
            st.session_state.pending_question = None
            st.warning("No relevant content found.")
        else:
            context = "\n".join(c["text"] for c in retrieved)
            prompt  = build_prompt(context, instruction, pq, chat_history=st.session_state.chat_history)

            # Generate (blocking — keeps layout clean)
            t_g0   = time.time()
            answer_parts = []
            for chunk in generate_answer_stream(prompt, max_new_tokens=300):
                answer_parts.append(chunk)
            answer = "".join(answer_parts)
            t_gen  = round((time.time() - t_g0) * 1000)

            # Faithfulness
            t_f0   = time.time()
            faith  = check_faithfulness(context, answer)
            t_faith = round((time.time() - t_f0) * 1000)
            t_total = round((time.time() - t_start) * 1000)

            # Save to history
            st.session_state.chat_history.append({
                "question": pq, "answer": answer, "faith": faith,
                "latency": t_total, "sources_count": len(retrieved),
                "sources": retrieved,
                "rewritten_query": rq if rq != pq else None,
            })
            st.session_state.chat_history   = st.session_state.chat_history[-5:]
            st.session_state.last_context   = context
            st.session_state.last_question  = pq
            st.session_state.latency_data   = {"retrieve":t_retrieve,"generate":t_gen,"faithful":t_faith,"total":t_total}
            st.session_state.is_processing  = False
            st.session_state.pending_question = None

            log_query(question=pq, rewritten_query=rq,
                      selected_doc=active_doc, num_chunks_retrieved=len(retrieved))

        # Rerun: now history has the new turn; it renders cleanly above the input
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Expandable details for the LAST completed turn
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.chat_history and not st.session_state.is_processing:
        last_turn = st.session_state.chat_history[-1]
        sources   = last_turn.get("sources", [])
        rq_used   = last_turn.get("rewritten_query")

        if rq_used:
            with st.expander("🔍 Search query used"):
                st.caption(rq_used)

        if sources:
            with st.expander(f"📌 View {len(sources)} retrieved sources"):
                for i, c in enumerate(sources, 1):
                    rerank = c.get("rerank_score", 0)
                    rrf    = c.get("score", 0)
                    stxt   = (f"RRF {rrf:.3f} · " if rrf > 0 else "") + f"Rerank {rerank:.2f}"
                    st.markdown(f"""
                    <div class="src-wrap">
                      <div class="src-hdr">
                        <div class="src-num">{i}</div>
                        <div class="src-file">{c['source']}</div>
                        <div class="src-scr">{stxt}</div>
                      </div>
                      <div class="src-text">{c['text'][:350]}{'...' if len(c['text'])>350 else ''}</div>
                    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Quick Actions (above input bar)
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.last_context and st.session_state.last_question and not st.session_state.is_processing:
        st.markdown('<hr class="main-divider">', unsafe_allow_html=True)
        st.markdown('<div class="qa-label">Quick Actions</div>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            if st.button("💡 Examples", use_container_width=True):
                with st.spinner("Generating..."):
                    out = generate_answer(build_prompt(
                        st.session_state.last_context,
                        "Give clear real-world examples based on the context.",
                        st.session_state.last_question,
                        chat_history=st.session_state.chat_history
                    ))
                st.markdown(f'<div class="bubble-asst" style="margin-top:8px;">{out}</div>', unsafe_allow_html=True)
        with f2:
            if st.button("📝 MCQs", use_container_width=True):
                with st.spinner("Generating..."):
                    out = generate_answer(build_prompt(
                        st.session_state.last_context,
                        get_instruction("2", 5),
                        st.session_state.last_question,
                        chat_history=st.session_state.chat_history
                    ))
                st.text_area("MCQs", value=out, height=180)
        with f3:
            if st.button("🧸 Simplify", use_container_width=True):
                with st.spinner("Simplifying..."):
                    out = generate_answer(build_prompt(
                        st.session_state.last_context,
                        "Explain this in very simple terms like explaining to a 5-year-old.",
                        st.session_state.last_question,
                        chat_history=st.session_state.chat_history
                    ))
                st.markdown(f'<div class="bubble-asst" style="margin-top:8px;">{out}</div>', unsafe_allow_html=True)
        with f4:
            if st.session_state.chat_history:
                last = st.session_state.chat_history[-1]
                ld   = st.session_state.latency_data or {}
                txt  = (f"Question: {last['question']}\n\nAnswer:\n{last['answer']}\n\n"
                        f"Document: {active_doc}\n"
                        + (f"Latency: {ld.get('total')}ms\n" if ld else ""))
                st.download_button("📥 Export", data=txt,
                    file_name="documind_answer.txt", mime="text/plain",
                    use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Input bar (always last in DOM = visually at bottom)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<hr class="main-divider">', unsafe_allow_html=True)
    icol1, icol2 = st.columns([5, 1])
    with icol1:
        placeholder = "Ask a follow-up..." if st.session_state.chat_history else "Ask anything about the document..."
        question = st.text_input(
            "Question", placeholder=placeholder,
            label_visibility="collapsed", key="question_input",
            disabled=st.session_state.is_processing
        )
    with icol2:
        ask = st.button(
            "⏳ Wait..." if st.session_state.is_processing else "Send →",
            type="primary", use_container_width=True,
            disabled=st.session_state.is_processing
        )
    st.caption("Press Enter or click Send · Answers appear above")

    # On send: set pending state and rerun — pipeline runs on next pass
    if (ask or question) and not st.session_state.is_processing:
        q = question.strip()
        if q:
            st.session_state.pending_question = q
            st.session_state.is_processing    = True
            st.rerun()