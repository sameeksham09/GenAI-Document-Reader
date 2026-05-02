# generator.py
from retriever import retrieve_context  # function that fetches relevant PDF chunks
from llm_utils import generate_answer
from doc_analyzer import analyze_document

# Example question
question = "What is DBMS?"

# Retrieve top-k relevant chunks from your indexed PDFs
chunks = retrieve_context(question, k=3)
context_text = "\n".join([c['text'] for c in chunks])

# Build prompt
prompt = f"Answer the question based on the following context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

# Generate answer (no device argument needed)
answer = generate_answer(prompt)
print("PROMPT:", question)
print("ANSWER:", answer)
