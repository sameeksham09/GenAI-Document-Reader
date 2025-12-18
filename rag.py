from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# Step 2: Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Step 3: Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save everything
faiss.write_index(index, "doc_index.faiss")
np.save("chunks.npy", np.array(chunks))

print("✅ Document indexed successfully.")
