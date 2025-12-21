import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import io
from PyPDF2 import PdfReader


# ---------------- CONFIG ----------------
SIMILARITY_THRESHOLD = 1.8
CHUNK_SIZE = 300

# ---------------- LOAD STATE ----------------
index = faiss.read_index("doc_index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- RETRIEVAL ----------------
def retrieve_context(question, k=4):
    """
    Retrieve top-k relevant chunks using FAISS with similarity thresholding
    """
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)

    D, I = index.search(question_embedding, k * 3)

    retrieved = []
    seen_ids = set()

    for score, idx in zip(D[0], I[0]):
        if score > SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[idx]

        if chunk["id"] in seen_ids:
            continue

        seen_ids.add(chunk["id"])

        retrieved.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "source": chunk.get("source", "notes.txt"),
            "score": float(score)
        })

        if len(retrieved) == k:
            break

    return retrieved

# ---------------- DOCUMENT INGESTION ----------------
def add_new_document(file_bytes, filename):
    """
    Add TXT or PDF documents dynamically to FAISS index
    """

    # 1️⃣ Extract text
    text = ""

    if filename.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8")

    elif filename.lower().endswith(".pdf"):
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "


    if not text.strip():
        return False

    # 2️⃣ Chunk
    words = text.split()
    new_chunks = []

    for i in range(0, len(words), CHUNK_SIZE):
        chunk = {
            "id": len(chunks),
            "text": " ".join(words[i:i + CHUNK_SIZE]),
            "source": filename
        }
        new_chunks.append(chunk)
        chunks.append(chunk)

    # 3️⃣ Embed
    embeddings = embed_model.encode([c["text"] for c in new_chunks])
    embeddings = np.array(embeddings, dtype=np.float32)

    # 4️⃣ Add to FAISS
    index.add(embeddings)

    # 5️⃣ Persist
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(index, "doc_index.faiss")

    return True
