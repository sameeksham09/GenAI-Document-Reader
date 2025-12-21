import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load FAISS index
index = faiss.read_index("doc_index.faiss")

# ✅ Load embedding model ONCE (THIS WAS MISSING)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(question, k=4):
    """
    Retrieve top-k relevant chunks along with similarity scores
    """
    # Convert question to embedding
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)

    # Search more than k to remove duplicates
    D, I = index.search(question_embedding, k * 4)

    seen_ids = set()
    retrieved = []

    for idx, distance in zip(I[0], D[0]):
        chunk_id = chunks[idx]["id"]

        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)

        retrieved.append({
            "id": chunk_id,
            "text": chunks[idx]["text"],
            "score": float(distance)
        })

        if len(retrieved) == k:
            break

    return retrieved
