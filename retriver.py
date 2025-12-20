import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load index and chunks once
index = faiss.read_index("doc_index.faiss")
chunks = np.load("chunks.npy", allow_pickle=True)

# Load embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(question, k=4):
    """
    Converts question to embedding and retrieves top-k relevant chunks
    """
    question_embedding = embed_model.encode([question])
    D, I = index.search(np.array(question_embedding), k)

    retrieved = [chunks[i] for i in I[0]]
    return retrieved
