import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load chunks (Pickle preserves IDs and text perfectly)
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load FAISS index
index = faiss.read_index("doc_index.faiss")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(question, k=4):
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)

    D, I = index.search(question_embedding, k * 4)

    print("Raw FAISS indices:", I[0])
    print("Raw FAISS distances:", D[0])
    print("Mapped chunk IDs:", [chunks[idx]["id"] for idx in I[0]])

    seen_ids = set()
    retrieved = []

    for idx in I[0]:
        chunk_id = chunks[idx]["id"]
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        retrieved.append({
            "id": chunk_id,
            "text": chunks[idx]["text"]
        })
        if len(retrieved) == k:
            break

    print("Returned unique chunk IDs:", [c["id"] for c in retrieved])
    return retrieved

