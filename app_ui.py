import streamlit as st
import pickle
import os

from retriver import retrieve_context, add_new_document
from prompts import get_instruction, build_prompt
from generator import generate_answer

DOC_META_FILE = "doc_metadata.pkl"

# -------------------------
# Page config
# -------------------------
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

            # Reload metadata after upload
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

# # ======================================================
# RIGHT COLUMN: Ask Questions + Output
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
            placeholder="Ask something from this document..."
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

                # Save for follow-ups
                st.session_state.last_context = context
                st.session_state.last_question = question

                # ✅ OUTPUT RIGHT HERE
                st.markdown("### 🤖 AI Answer")
                st.text_area(
                    "Response",
                    value=answer,
                    height=250
                )

                st.markdown("### 📌 Sources")
                for c in retrieved:
                    st.write(f"- {c['source']} | similarity: {c['score']:.4f}")

    # ==================================================
    # SMART FOLLOW-UPS
    # ==================================================
    if st.session_state.last_context and st.session_state.last_question:
        st.subheader("🔁 Smart Follow-ups")
        st.caption("Would you like:")

        f1, f2, f3 = st.columns(3)

        with f1:
            if st.button("1️⃣ Examples"):
                instruction = "Give clear, real-world examples based on the context."
                prompt = build_prompt(
                    st.session_state.last_context,
                    instruction,
                    st.session_state.last_question
                )
                output = generate_answer(prompt)
                st.text_area("📘 Examples", value=output, height=250)

        with f2:
            if st.button("2️⃣ MCQs"):
                instruction = get_instruction("2", 5)
                prompt = build_prompt(
                    st.session_state.last_context,
                    instruction,
                    st.session_state.last_question
                )
                output = generate_answer(prompt)
                st.text_area("📝 MCQs", value=output, height=250)

        with f3:
            if st.button("3️⃣ Explain like I'm 5"):
                instruction = "Explain this in very simple terms like explaining to a 5-year-old."
                prompt = build_prompt(
                    st.session_state.last_context,
                    instruction,
                    st.session_state.last_question
                )
                output = generate_answer(prompt)
                st.text_area("🧸 ELI5", value=output, height=250)
