import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and chunks
index = faiss.read_index("doc_index.faiss")
chunks = np.load("chunks.npy", allow_pickle=True)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("📄 RAG system ready. Ask questions based on the document.")

print("""
Choose question type:
1 - Descriptive
2 - MCQ
3 - True / False
4 - Fill in the blanks
""")

def normalize_question(q):
    q = q.lower().strip()
    if not q.endswith("?"):
        q = q + "?"
    return q


while True:
    qtype = input("\nEnter question type (1/2/3/4 or exit): ").strip()
    if qtype.lower() == "exit":
        break

    raw_question = input("Enter your question: ")
    question = normalize_question(raw_question)


    # Embed question
    question_embedding = embed_model.encode([question])
    D, I = index.search(np.array(question_embedding), k=4)
    retrieved_chunks = [chunks[i] for i in I[0]]

    if not retrieved_chunks:
        print("🤖 AI Answer: I don't know based on the provided document.")
        continue

    context = "\n".join(retrieved_chunks)

    # Prompt templates
    if qtype == "1":
        instruction = "Answer the question clearly."
    elif qtype == "2":
        instruction = "Create 4 multiple-choice options and clearly mark the correct answer."
    elif qtype == "3":
        instruction = "Answer strictly as True or False and give one-line justification."
    elif qtype == "4":
        instruction = "Create a fill-in-the-blank question and provide the answer."
    else:
        print("Invalid option.")
        continue

    style_guard = """
Do not explain concepts beyond the document.
Do not introduce new definitions.
"""

    # Step 4: Send ONLY retrieved chunks to LLM
    prompt = f"""
Use ONLY the context below.
Do NOT use prior knowledge.
If the answer cannot be directly inferred from the context, say:
"I don't know based on the provided document."


Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n🤖 AI Output:")
    print(response["message"]["content"])
