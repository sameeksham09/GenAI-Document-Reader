import streamlit as st
from retriver import retrieve_context
from prompts import get_instruction, build_prompt
from generator import generate_answer

st.set_page_config(page_title="📄 RAG QA System", layout="wide")

st.title("📄 RAG QA System")
st.markdown("Ask questions based on your document and get grounded answers with citations.")

# Question type selection
qtype = st.selectbox(
    "Select question type:",
    ["Descriptive", "MCQ", "True / False", "Fill in the blanks"]
)

# Map selection to number for get_instruction()
qtype_mapping = {
    "Descriptive": "1",
    "MCQ": "2",
    "True / False": "3",
    "Fill in the blanks": "4"
}
qtype_number = qtype_mapping[qtype]

# Input question
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question!")
    else:
        instruction = get_instruction(qtype_number)

        # Retrieve relevant chunks from RAG retriever
        retrieved_chunks = retrieve_context(question)

        if not retrieved_chunks:
            st.write("🤖 AI Output: I don't know based on the provided document.")
        else:
            # Build context and prompt
            context = "\n".join([c["text"] for c in retrieved_chunks])
            prompt = build_prompt(context, instruction, question)
            answer = generate_answer(prompt)

            # Display AI answer
            st.subheader("🤖 AI Output:")
            st.write(answer)

            # Display citations with similarity scores
            st.subheader("📌 Sources & Similarity Scores:")
            for c in retrieved_chunks:
                st.write(f"- notes.txt | chunk {c['id']} | similarity score: {c['score']:.4f}")
