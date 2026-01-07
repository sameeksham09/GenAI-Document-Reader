import streamlit as st
import pickle
import os

from retriver import retrieve_context, add_new_document
from prompts import get_instruction, build_prompt
from generator import generate_answer

DOC_LIST_FILE = "uploaded_docs.pkl"
DOC_META_FILE = "doc_metadata.pkl"

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄 RAG QA System")
st.markdown("Ask questions based on your documents and get grounded answers with citations.")

# -------------------------
# CREATE TWO COLUMNS
# -------------------------
col1, col2 = st.columns(2)

# -------------------------
# LEFT COLUMN: Upload + Summary
# -------------------------
with col1:
    st.subheader("Upload a new document (TXT or PDF)")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

    current_file = None

    if uploaded_file:
        with st.spinner("Processing and adding document..."):
            added = add_new_document(uploaded_file.read(), uploaded_file.name)

        if added:
            st.success(f"✅ {uploaded_file.name} added successfully!")
            current_file = uploaded_file.name

    # -------------------------
    # Knowledge Base Documents
    # -------------------------
    st.subheader("📂 Knowledge Base Documents")

    if os.path.exists(DOC_LIST_FILE):
        with open(DOC_LIST_FILE, "rb") as f:
            uploaded_docs = pickle.load(f)
    else:
        uploaded_docs = []

    if not uploaded_docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in uploaded_docs:
            if doc == current_file:
                st.write(f"📄 **{doc}** (new)")
            else:
                st.write(f"📄 {doc}")

    # -------------------------
    # Document Summary
    # -------------------------
    st.subheader("📄 Document Summary / Insights")

    # Load metadata
    if os.path.exists(DOC_META_FILE):
        with open(DOC_META_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = {}

    if not metadata:
        st.info("No document summaries available yet.")
    else:
        doc_names = list(metadata.keys())
        placeholder = "— Select a document —"
        options = [placeholder] + doc_names

        # Default to placeholder, only show summary when a real document is selected
        selected_doc = st.selectbox(
            "Select document",
            options=options,
            index=0
        )

        if selected_doc != placeholder:
            summary = metadata[selected_doc].get("summary", "No summary available.")
            st.markdown("### Summary")
            st.write(summary)


# -------------------------
# RIGHT COLUMN: Ask Questions
# -------------------------
with col2:
    st.subheader("Ask Questions")

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
            min_value=1, max_value=20, value=5
        )
    else:
        num_questions = 1

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        instruction = get_instruction(qtype_number, num_questions)
        retrieved = retrieve_context(question, selected_doc=selected_doc)

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
