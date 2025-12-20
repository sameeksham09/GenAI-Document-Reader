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
    """
    Converts question to embedding and retrieves top-k relevant chunks
    """
    # Convert question to embedding
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)  # FAISS requires float32

    # Search top-k nearest chunks
    D, I = index.search(question_embedding, k)

    # Map indices to chunks
    retrieved = []
    for idx in I[0]:
        retrieved.append({
            "id": chunks[idx]["id"],
            "text": chunks[idx]["text"]
        })

    return retrieved
