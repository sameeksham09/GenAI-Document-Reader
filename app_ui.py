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
  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Subtle warm off-white background ── */
  .stApp { background: #F7F6F3; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8E6E1;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
  }

  /* ── Remove default block padding ── */
  .block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

  /* ── Logo area ── */
  .logo-wrap {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 0 16px;
  }
  .logo-box {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #7B6EF6, #A78BFA);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    box-shadow: 0 2px 8px rgba(123,110,246,0.3);
  }
  .logo-text { font-size: 17px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.02em; }
  .logo-sub  { font-size: 11px; color: #aaa; margin-top: 1px; }

  /* ── Section labels ── */
  .sec-label {
    font-size: 10px; font-weight: 600;
    color: #bbb; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 8px;
    padding: 0 2px;
  }

  /* ── Document cards ── */
  .doc-card-active {
    background: linear-gradient(135deg, #F0EFFE, #EAE7FD);
    border: 1px solid #C4BCFB;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 6px;
    box-shadow: 0 1px 4px rgba(123,110,246,0.12);
  }
  .doc-card-inactive {
    background: #FAFAF9;
    border: 1px solid #E8E6E1;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 6px;
    transition: all 0.15s;
  }
  .doc-name-active   { font-size: 12px; font-weight: 600; color: #4338CA; line-height: 1.4; }
  .doc-name-inactive { font-size: 12px; font-weight: 500; color: #374151; line-height: 1.4; }
  .doc-meta-active   { font-size: 11px; color: #7C70E8; margin-top: 3px; }
  .doc-meta-inactive { font-size: 11px; color: #9CA3AF; margin-top: 3px; }

  /* ── Stats cards ── */
  .stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 4px 0 12px; }
  .stat-box {
    background: #F7F6F3;
    border: 1px solid #E8E6E1;
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
  }
  .stat-n { font-size: 20px; font-weight: 600; color: #1a1a1a; }
  .stat-l { font-size: 10px; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 1px; }

  /* ── Thin divider ── */
  .divider { border: none; border-top: 1px solid #E8E6E1; margin: 12px 0; }

  /* ── Empty state ── */
  .empty-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 65vh; text-align: center;
  }
  .empty-icon { font-size: 52px; margin-bottom: 16px; opacity: 0.35; }
  .empty-h    { font-size: 26px; font-weight: 600; color: #374151; margin-bottom: 8px; letter-spacing: -0.03em; }
  .empty-p    { font-size: 14px; color: #9CA3AF; max-width: 340px; line-height: 1.6; }
  .badge-row  { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 20px; }
  .badge {
    background: #FFFFFF;
    border: 1px solid #E8E6E1;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px; color: #6B7280;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }

  /* ── Main content card ── */
  .main-card {
    background: #FFFFFF;
    border: 1px solid #E8E6E1;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 16px;
  }

  /* ── Topbar doc info ── */
  .topbar-name { font-size: 15px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.01em; }
  .topbar-sub  { font-size: 11px; color: #9CA3AF; margin-top: 2px; }

  /* ── Chat bubbles ── */
  .user-row {
    display: flex; flex-direction: column;
    align-items: flex-end; margin: 12px 0;
  }
  .assistant-row {
    display: flex; flex-direction: column;
    align-items: flex-start; margin: 12px 0;
  }
  .bubble-lbl {
    font-size: 10px; font-weight: 600; color: #9CA3AF;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 5px;
  }
  .bubble-lbl-r {
    font-size: 10px; font-weight: 600; color: #9CA3AF;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 5px; text-align: right;
  }
  .bubble-user {
    background: linear-gradient(135deg, #7B6EF6, #9D8FFF);
    color: #ffffff;
    border-radius: 14px 4px 14px 14px;
    padding: 11px 16px;
    font-size: 14px; line-height: 1.6;
    max-width: 76%;
    box-shadow: 0 2px 8px rgba(123,110,246,0.25);
  }
  .bubble-asst {
    background: #FFFFFF;
    border: 1px solid #E8E6E1;
    color: #1a1a1a;
    border-radius: 4px 14px 14px 14px;
    padding: 11px 16px;
    font-size: 14px; line-height: 1.65;
    max-width: 82%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }

  /* ── Meta pills ── */
  .pill-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; }
  .pill-faithful   { background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; border-radius:20px; padding:2px 9px; font-size:11px; font-weight:500; }
  .pill-partial    { background:#FFFBEB; color:#92400E; border:1px solid #FCD34D; border-radius:20px; padding:2px 9px; font-size:11px; font-weight:500; }
  .pill-unfaithful { background:#FEF2F2; color:#991B1B; border:1px solid #FCA5A5; border-radius:20px; padding:2px 9px; font-size:11px; font-weight:500; }
  .pill-neutral    { background:#F3F4F6; color:#6B7280; border:1px solid #E5E7EB; border-radius:20px; padding:2px 9px; font-size:11px; }

  /* ── Source cards ── */
  .src-card {
    background: #FAFAF9;
    border: 1px solid #E8E6E1;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
  }
  .src-hdr  { display:flex; align-items:center; gap:7px; margin-bottom:6px; }
  .src-rank {
    background: linear-gradient(135deg, #7B6EF6, #A78BFA);
    color: white; width:20px; height:20px;
    border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:10px; font-weight:700;
    flex-shrink:0;
  }
  .src-file  { font-size:11px; color:#9CA3AF; flex:1; }
  .src-score { font-size:11px; color:#7B6EF6; font-weight:500; }
  .src-text  { font-size:12px; color:#4B5563; line-height:1.55; }

  /* ── Input area ── */
  .stTextInput input {
    border-radius: 10px !important;
    border: 1.5px solid #E8E6E1 !important;
    background: #FAFAF9 !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s !important;
  }
  .stTextInput input:focus {
    border-color: #7B6EF6 !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(123,110,246,0.08) !important;
  }

  /* ── Primary button ── */
  .stButton button[kind="primary"] {
    background: linear-gradient(135deg, #7B6EF6, #9D8FFF) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(123,110,246,0.3) !important;
    transition: all 0.2s !important;
  }

  /* ── Sidebar buttons ── */
  .stButton button {
    border-radius: 8px !important;
    font-size: 12px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "selected_doc": None, "last_context": None, "last_question": None,
    "chat_history": [], "last_rewritten_query": None,
    "confirm_delete": False, "latency_data": None,
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

    # Logo
    st.markdown("""
    <div class="logo-wrap">
      <div class="logo-box">🧠</div>
      <div>
        <div class="logo-text">DocuMind</div>
        <div class="logo-sub">RAG · Document Q&A</div>
      </div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="sec-label">Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF or TXT", type=["txt","pdf"], label_visibility="collapsed"
    )
    if uploaded_file:
        with st.spinner("Indexing..."):
            added = add_new_document(uploaded_file.read(), uploaded_file.name)
        if added:
            st.success("✅ Indexed successfully!")
            st.session_state.selected_doc = uploaded_file.name
            metadata = load_metadata()
        else:
            st.session_state.selected_doc = uploaded_file.name
            if uploaded_file.name in get_all_documents():
                st.info("Already indexed.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Documents
    st.markdown('<div class="sec-label">Documents</div>', unsafe_allow_html=True)
    doc_list = get_all_documents()

    if not doc_list:
        st.markdown('<div style="font-size:12px;color:#bbb;padding:4px 2px;">No documents yet. Upload one above.</div>', unsafe_allow_html=True)
    else:
        from retriever import collection as _col
        for doc in doc_list:
            is_active   = doc == st.session_state.selected_doc
            res         = _col.get(where={"source": doc})
            chunk_count = len(res["ids"]) if res["ids"] else 0
            all_logs    = load_logs(last_n=10000)
            qcount      = sum(1 for l in all_logs if l.get("event") == "query" and l.get("doc") == doc)
            short_name  = doc[:26] + "..." if len(doc) > 26 else doc

            card  = "doc-card-active"  if is_active else "doc-card-inactive"
            ncls  = "doc-name-active"  if is_active else "doc-name-inactive"
            mcls  = "doc-meta-active"  if is_active else "doc-meta-inactive"

            st.markdown(f"""
            <div class="{card}">
              <div class="{ncls}">{short_name}</div>
              <div class="{mcls}">🧩 {chunk_count} chunks &nbsp;·&nbsp; ❓ {qcount} queries</div>
            </div>
            """, unsafe_allow_html=True)

            if not is_active:
                if st.button("Select", key=f"sel_{doc}", use_container_width=True):
                    st.session_state.selected_doc  = doc
                    st.session_state.chat_history  = []
                    st.session_state.last_context  = None
                    st.session_state.last_question = None
                    st.session_state.latency_data  = None
                    st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Delete
    if st.session_state.selected_doc and st.session_state.selected_doc in doc_list:
        if not st.session_state.confirm_delete:
            if st.button("🗑 Delete active document", use_container_width=True):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning(f"Delete **{st.session_state.selected_doc[:22]}**?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete", use_container_width=True):
                    delete_document(st.session_state.selected_doc)
                    st.session_state.selected_doc   = None
                    st.session_state.chat_history   = []
                    st.session_state.confirm_delete = False
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Stats
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-n">{stats['total_uploads']}</div>
        <div class="stat-l">Docs</div>
      </div>
      <div class="stat-box">
        <div class="stat-n">{stats['total_queries']}</div>
        <div class="stat-l">Queries</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.chat_history         = []
        st.session_state.last_context         = None
        st.session_state.last_question        = None
        st.session_state.last_rewritten_query = None
        st.session_state.latency_data         = None
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
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.selected_doc:
    # Empty state
    st.markdown("""
    <div class="empty-wrap">
      <div class="empty-icon">🧠</div>
      <div class="empty-h">DocuMind</div>
      <div class="empty-p">Upload a document from the sidebar, then ask anything about its contents.</div>
      <div class="badge-row">
        <span class="badge">🔍 Hybrid BM25 + Dense</span>
        <span class="badge">⚡ Cross-Encoder Reranking</span>
        <span class="badge">💬 Conversation Memory</span>
        <span class="badge">🛡 Faithfulness Check</span>
        <span class="badge">🤖 LLaMA 3.2</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    active_doc = st.session_state.selected_doc

    # Topbar
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.markdown(f"""
        <div style="padding:4px 0 12px;">
          <div class="topbar-name">📄 {active_doc}</div>
          <div class="topbar-sub">Active document · Ask anything about its contents</div>
        </div>
        """, unsafe_allow_html=True)
    with tcol2:
        qtype = st.selectbox(
            "Type", ["Descriptive","MCQ","True / False","Fill in the blanks"],
            label_visibility="collapsed"
        )
        num_q = st.number_input(
            "Count", 1, 20, 5, label_visibility="collapsed"
        ) if qtype in ["MCQ","True / False"] else 1

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Conversation history
    if st.session_state.chat_history:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        for turn in st.session_state.chat_history:
            st.markdown(f"""
            <div class="user-row">
              <div class="bubble-lbl-r">You</div>
              <div class="bubble-user">{turn['question']}</div>
            </div>
            """, unsafe_allow_html=True)

            sc = turn.get("faith",{}).get("score",5)
            rs = turn.get("faith",{}).get("reason","")
            if sc >= 7:
                fpill = f'<span class="pill-faithful">✓ Faithful {sc}/10</span>'
            elif sc >= 4:
                fpill = f'<span class="pill-partial">~ Partial {sc}/10</span>'
            else:
                fpill = f'<span class="pill-unfaithful">✗ Not faithful {sc}/10</span>'

            lat   = f'<span class="pill-neutral">⚡ {turn["latency"]}ms</span>' if turn.get("latency") else ""
            srcs  = f'<span class="pill-neutral">📌 {turn["sources_count"]} sources</span>' if turn.get("sources_count") else ""

            st.markdown(f"""
            <div class="assistant-row">
              <div class="bubble-lbl">🧠 DocuMind</div>
              <div class="bubble-asst">{turn['answer']}</div>
              <div class="pill-row">{fpill}{lat}{srcs}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input
    icol1, icol2 = st.columns([5, 1])
    with icol1:
        question = st.text_input(
            "Question",
            placeholder="Ask anything about the document...",
            label_visibility="collapsed"
        )
    with icol2:
        ask = st.button("Send →", type="primary", use_container_width=True)

    st.caption("Press Enter or click Send · Answers stream in real time")

    # Pipeline
    if ask and question.strip():
        qtype_map    = {"Descriptive":"1","MCQ":"2","True / False":"3","Fill in the blanks":"4"}
        instruction  = get_instruction(qtype_map[qtype], num_q)
        t_start      = time.time()

        # Query rewriting
        if st.session_state.chat_history:
            with st.spinner("Resolving context..."):
                rp = build_rewrite_prompt(question, st.session_state.chat_history)
                rq = generate_answer(rp, max_new_tokens=80)
                lines = [l.strip() for l in rq.splitlines() if l.strip()]
                rq    = lines[0] if lines else question
                for pfx in ["Rewritten query:","Rewritten:","Query:","Question:"]:
                    if rq.lower().startswith(pfx.lower()):
                        rq = rq[len(pfx):].strip()
                rq = rq.strip('"\'')
                if len(rq) < 5: rq = question
            st.session_state.last_rewritten_query = rq
        else:
            rq = question
            st.session_state.last_rewritten_query = None

        # Retrieval
        t_r0 = time.time()
        with st.spinner("Searching document..."):
            retrieved = retrieve_context(rq, selected_doc=active_doc)
        t_retrieve = round((time.time() - t_r0) * 1000)

        if not retrieved:
            st.warning("No relevant content found.")
        else:
            context = "\n".join(c["text"] for c in retrieved)
            prompt  = build_prompt(context, instruction, question,
                                   chat_history=st.session_state.chat_history)

            # User bubble
            st.markdown(f"""
            <div class="main-card" style="padding:14px 18px;">
              <div class="user-row" style="margin:0;">
                <div class="bubble-lbl-r">You</div>
                <div class="bubble-user">{question}</div>
              </div>
            """, unsafe_allow_html=True)

            # Answer stream
            st.markdown('<div class="assistant-row" style="margin:12px 0 0;"><div class="bubble-lbl">🧠 DocuMind</div>', unsafe_allow_html=True)
            t_g0   = time.time()
            answer = st.write_stream(generate_answer_stream(prompt, max_new_tokens=300))
            st.markdown('</div>', unsafe_allow_html=True)
            t_gen  = round((time.time() - t_g0) * 1000)

            # Faithfulness
            t_f0 = time.time()
            with st.spinner("Checking faithfulness..."):
                faith = check_faithfulness(context, answer)
            t_faith = round((time.time() - t_f0) * 1000)
            t_total = round((time.time() - t_start) * 1000)

            sc = faith["score"]
            rs = faith["reason"]
            if sc >= 7:
                fpill = f'<span class="pill-faithful">✓ Faithful {sc}/10</span>'
            elif sc >= 4:
                fpill = f'<span class="pill-partial">~ Partial {sc}/10</span>'
            else:
                fpill = f'<span class="pill-unfaithful">✗ Not faithful {sc}/10</span>'

            st.markdown(f"""
              <div class="pill-row">
                {fpill}
                <span class="pill-neutral">⚡ {t_total}ms</span>
                <span class="pill-neutral">📌 {len(retrieved)} sources</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Rewrite
            if st.session_state.last_rewritten_query and st.session_state.last_rewritten_query != question:
                with st.expander("🔍 Search query used"):
                    st.caption(st.session_state.last_rewritten_query)

            # Sources
            with st.expander(f"📌 View {len(retrieved)} retrieved sources"):
                for i, c in enumerate(retrieved, 1):
                    rerank = c.get("rerank_score", 0)
                    rrf    = c.get("score", 0)
                    stxt   = f"RRF {rrf:.3f} · " if rrf > 0 else ""
                    stxt  += f"Rerank {rerank:.2f}"
                    st.markdown(f"""
                    <div class="src-card">
                      <div class="src-hdr">
                        <div class="src-rank">{i}</div>
                        <div class="src-file">{c['source']}</div>
                        <div class="src-score">{stxt}</div>
                      </div>
                      <div class="src-text">{c['text'][:350]}{'...' if len(c['text'])>350 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Save
            st.session_state.last_context  = context
            st.session_state.last_question = question
            st.session_state.chat_history.append({
                "question": question, "answer": answer,
                "faith": faith, "latency": t_total, "sources_count": len(retrieved),
            })
            st.session_state.chat_history  = st.session_state.chat_history[-5:]
            st.session_state.latency_data  = {"retrieve":t_retrieve,"generate":t_gen,"faithful":t_faith,"total":t_total}

            log_query(question=question, rewritten_query=rq,
                      selected_doc=active_doc, num_chunks_retrieved=len(retrieved))

    # Follow-ups
    if st.session_state.last_context and st.session_state.last_question:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;color:#bbb;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">Quick Actions</div>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            if st.button("💡 Examples", use_container_width=True):
                with st.spinner("Generating..."):
                    out = generate_answer(build_prompt(
                        st.session_state.last_context,
                        "Give clear, real-world examples based on the context.",
                        st.session_state.last_question,
                        chat_history=st.session_state.chat_history
                    ))
                st.markdown(f'<div class="main-card"><div class="bubble-lbl">💡 Examples</div><div class="bubble-asst">{out}</div></div>', unsafe_allow_html=True)

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
                st.markdown(f'<div class="main-card"><div class="bubble-lbl">🧸 Simple</div><div class="bubble-asst">{out}</div></div>', unsafe_allow_html=True)

        with f4:
            if st.session_state.chat_history:
                last = st.session_state.chat_history[-1]
                ld   = st.session_state.latency_data or {}
                txt  = (f"Question: {last['question']}\n\nAnswer:\n{last['answer']}\n\n"
                        f"Document: {active_doc}\n"
                        + (f"Total latency: {ld.get('total')}ms\n" if ld else ""))
                st.download_button(
                    "📥 Export", data=txt,
                    file_name="documind_answer.txt",
                    mime="text/plain", use_container_width=True
                )