import streamlit as st
from retriver import retrieve_context, add_new_document
from prompts import get_instruction, build_prompt
from generator import generate_answer

st.set_page_config(page_title="📄 RAG QA System", layout="wide")
st.title("📄 RAG QA System")
st.markdown(
    "Ask questions based on your documents and get grounded answers with citations."
)

# -------------------------
# 1️⃣ File Upload
# -------------------------
st.subheader("Upload a new document (TXT or PDF)")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    add_new_document(file_bytes, filename)
    st.success(f"✅ {filename} has been added to the knowledge base!")

# -------------------------
# 2️⃣ Question Type
# -------------------------
qtype = st.selectbox(
    "Select question type:",
    ["Descriptive", "MCQ", "True / False", "Fill in the blanks"]
)

qtype_mapping = {
    "Descriptive": "1",
    "MCQ": "2",
    "True / False": "3",
    "Fill in the blanks": "4"
}
qtype_number = qtype_mapping[qtype]

if qtype == "MCQ":
    num_questions = st.number_input(
        "How many MCQs do you want?",
        min_value=1, max_value=20, value=5, step=1
    )
elif qtype == "True / False":
    num_questions = st.number_input(
        "How many True/False questions do you want?",
        min_value=1, max_value=20, value=5, step=1
    )
else:
    num_questions = 1    

# -------------------------
# 3️⃣ Dynamic Placeholder
# -------------------------
question_placeholder = {
    "Descriptive": "Type your question here...",
    "MCQ": "Enter the topic or question for MCQ generation...",
    "True / False": "Enter a statement to verify True or False...",
    "Fill in the blanks": "Enter a sentence or concept for fill-in-the-blank..."
}

question = st.text_input(
    "Enter your question:",
    placeholder=question_placeholder[qtype]
)

# -------------------------
# 4️⃣ Get Answer
# -------------------------
if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question!")
    else:
        instruction = get_instruction(qtype_number, num_questions)

        # Retrieve relevant chunks
        retrieved_chunks = retrieve_context(question)

        if not retrieved_chunks:
            st.write("🤖 AI Output: I don't know based on the provided document.")
        else:
            # Build context
            context = "\n".join([c["text"] for c in retrieved_chunks])
            prompt = build_prompt(context, instruction, question)

            # Generate answer
            answer = generate_answer(prompt)

            # Display answer
            st.subheader("🤖 AI Output:")
            st.write(answer)

            # Display sources & similarity scores
            st.subheader("📌 Sources & Similarity Scores:")
            for c in retrieved_chunks:
                st.write(f"- {c['source']} | similarity score: {c['score']:.4f}")
