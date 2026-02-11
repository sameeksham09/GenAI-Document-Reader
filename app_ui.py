import streamlit as st
import pickle
import os

from retriver import retrieve_context, add_new_document
from prompts import get_instruction, build_prompt
from generator import generate_answer

DOC_META_FILE = "doc_metadata.pkl"

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄 RAG QA System")
st.markdown("Ask questions based on your documents and get grounded answers with citations.")

# -------------------------
# Session State Init
# -------------------------
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

if "last_context" not in st.session_state:
    st.session_state.last_context = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

# -------------------------
# Load Metadata
# -------------------------
if os.path.exists(DOC_META_FILE):
    with open(DOC_META_FILE, "rb") as f:
        metadata = pickle.load(f)
else:
    metadata = {}

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns(2)

# ======================================================
# LEFT COLUMN: Upload + Summary
# ======================================================
with col1:
    st.subheader("Upload a new document (TXT or PDF)")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

    if uploaded_file:
        with st.spinner("Processing and adding document..."):
            added = add_new_document(uploaded_file.read(), uploaded_file.name)

        if added:
            st.success(f"✅ {uploaded_file.name} added successfully!")
            st.session_state.selected_doc = uploaded_file.name

            if os.path.exists(DOC_META_FILE):
                with open(DOC_META_FILE, "rb") as f:
                    metadata = pickle.load(f)

    st.subheader("📄 Document Summary / Insights")

    if not st.session_state.selected_doc:
        st.info("Upload a document to view its summary.")
    else:
        doc = st.session_state.selected_doc
        summary = metadata.get(doc, {}).get("summary", "No summary available.")
        st.markdown(summary)

# ======================================================
# RIGHT COLUMN: Questions + Follow-ups
# ======================================================
with col2:
    st.subheader("Ask Questions")

    if not st.session_state.selected_doc:
        st.info("Please upload a document before asking questions.")
    else:
        active_doc = st.session_state.selected_doc

        qtype = st.selectbox(
            "Select question type:",
            ["Descriptive", "MCQ", "True / False", "Fill in the blanks"]
        )

        qtype_map = {
            "Descriptive": "1",
            "MCQ": "2",
            "True / False": "3",
            "Fill in the blanks": "4"
        }
        qtype_number = qtype_map[qtype]

        if qtype in ["MCQ", "True / False"]:
            num_questions = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=20,
                value=5
            )
        else:
            num_questions = 1

        question = st.text_input(
            "Enter your question:",
            placeholder="Ask about this document or request quiz-style questions."
        )

        if st.button("Get Answer"):
            instruction = get_instruction(qtype_number, num_questions)

            retrieved = retrieve_context(
                question,
                selected_doc=active_doc
            )

            if not retrieved:
                st.warning("I don't know based on the provided document.")
            else:
                context = "\n".join(c["text"] for c in retrieved)
                prompt = build_prompt(context, instruction, question)
                answer = generate_answer(prompt)

                st.subheader("🤖 AI Output")
                st.text_area("Answer", value=answer, height=300)

                st.subheader("📌 Sources")
                for c in retrieved:
                    st.write(f"- {c['source']} | similarity: {c['score']:.4f}")
