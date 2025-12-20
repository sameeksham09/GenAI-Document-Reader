from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load document
with open("notes.txt", "r") as f:
    text = f.read()

# Step 1: Chunk the document
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(text)

# Step 2: Create chunks with IDs
chunks_with_ids = [{"id": i, "text": chunks[i]} for i in range(len(chunks))]

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Create embeddings
embeddings = model.encode([c["text"] for c in chunks_with_ids])
embeddings = np.array(embeddings, dtype=np.float32)  # FAISS requires float32

# Step 5: Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 6: Save everything
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks_with_ids, f)

faiss.write_index(index, "doc_index.faiss")

print("✅ Document indexed successfully.")
