import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

SIMILARITY_THRESHOLD = 1.8  # 🔥 key addition

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load FAISS index
index = faiss.read_index("doc_index.faiss")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(question, k=4):
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)

    D, I = index.search(question_embedding, k)

    print("\n🔍 Retrieved Chunks:")
    retrieved = []

    for score, idx in zip(D[0], I[0]):
        safe_score = float(score)
        print(f"- Chunk {chunks[idx]['id']} | score: {safe_score:.4f}")


        if score > SIMILARITY_THRESHOLD:
            continue  # ❌ reject weak matches

        retrieved.append({
            "id": chunks[idx]["id"],
            "text": chunks[idx]["text"],
            "score": score
        })

    return retrieved
