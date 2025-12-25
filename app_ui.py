import streamlit as st
import pickle
from retriver import retrieve_context, add_new_document
from prompts import get_instruction, build_prompt
from generator import generate_answer

DOC_LIST_FILE = "uploaded_docs.pkl"
DOC_META_FILE = "doc_metadata.pkl"

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄 RAG QA System")
st.markdown("Ask questions based on your documents and get grounded answers with citations.")

# -------------------------
# Upload Document
# -------------------------
st.subheader("Upload a new document (TXT or PDF)")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

if uploaded_file:
    added = add_new_document(uploaded_file.read(), uploaded_file.name)
    if added:
        st.success(f"✅ {uploaded_file.name} added successfully!")

# -------------------------
# Knowledge Base
# -------------------------
st.subheader("📂 Knowledge Base Documents")
try:
    with open(DOC_LIST_FILE, "rb") as f:
        docs = pickle.load(f)
    if docs:
        for doc in docs:
            st.write(f"- {doc}")
    else:
        st.info("No documents uploaded yet.")
except:
    st.info("No documents uploaded yet.")

# -------------------------
# Document Summary
# -------------------------
try:
    with open(DOC_LIST_FILE, "rb") as f:
        uploaded_docs = pickle.load(f)
except:
    uploaded_docs = []

try:
    with open(DOC_META_FILE, "rb") as f:
        metadata = pickle.load(f)
except:
    metadata = {}

if uploaded_docs:
    st.subheader("📄 Document Summary / Insights")

    for doc in uploaded_docs:
        if doc in metadata:
            with st.expander(doc):
                st.write(metadata[doc])
else:
    st.info("No documents uploaded yet.")


# -------------------------
# Question Mode
# -------------------------
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

# -------------------------
# Ask
# -------------------------
if st.button("Get Answer"):
    instruction = get_instruction(qtype_number, num_questions)
    retrieved = retrieve_context(question)

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
