import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import io
from PyPDF2 import PdfReader
from doc_analyzer import analyze_document

# ---------------- CONFIG ----------------
SIMILARITY_THRESHOLD = 1.8
CHUNK_SIZE = 300

CHUNKS_FILE = "chunks.pkl"
INDEX_FILE = "doc_index.faiss"
DOC_LIST_FILE = "uploaded_docs.pkl"
DOC_META_FILE = "doc_metadata.pkl"

# ---------------- LOAD / INIT STATE ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load or initialize FAISS index
try:
    index = faiss.read_index(INDEX_FILE)
except:
    index = faiss.IndexFlatL2(384)  # MiniLM embedding size

# Load or initialize chunks
try:
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
except:
    chunks = []

# ---------------- RETRIEVAL ----------------
def retrieve_context(question, k=4, selected_doc=None):
    question_embedding = embed_model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)

    if index.ntotal == 0:
        print("FAISS index empty. No documents to retrieve from.")
        return []


    D, I = index.search(question_embedding, k * 3)

    retrieved = []
    seen = set()

    for score, idx in zip(D[0], I[0]):
        if score > SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[idx]
        if chunk["id"] in seen:
            continue

        seen.add(chunk["id"])

        # Only keep chunks from the selected document if provided
        if selected_doc and chunk["source"] != selected_doc:
            continue

        retrieved.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "score": float(score)
        })

        if len(retrieved) == k:
            break

    return retrieved


# ---------------- DOCUMENT INGESTION ----------------
def add_new_document(file_bytes, filename):
    # 1️⃣ Extract text
    text = ""

    if filename.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")

    elif filename.lower().endswith(".pdf"):
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "

    if not text.strip():
        return False

    # 2️⃣ Chunk text
    words = text.split()
    new_chunks = []

    for i in range(0, len(words), CHUNK_SIZE):
        chunk = {
            "id": len(chunks),
            "text": " ".join(words[i:i + CHUNK_SIZE]),
            "source": filename
        }
        chunks.append(chunk)
        new_chunks.append(chunk)

    # 3️⃣ Embed
    embeddings = embed_model.encode([c["text"] for c in new_chunks])
    embeddings = np.array(embeddings, dtype=np.float32)

    index.add(embeddings)

    # 4️⃣ Persist FAISS + chunks
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, INDEX_FILE)

    # 5️⃣ Track uploaded documents
    try:
        with open(DOC_LIST_FILE, "rb") as f:
            uploaded_docs = pickle.load(f)
    except:
        uploaded_docs = []

    if filename not in uploaded_docs:
        uploaded_docs.append(filename)

    with open(DOC_LIST_FILE, "wb") as f:
        pickle.dump(uploaded_docs, f)

    # 6️⃣ Analyze document (ChatPDF-style)
    analysis = analyze_document(text)

    # Ensure metadata is always a dict
    try:
        with open(DOC_META_FILE, "rb") as f:
            metadata = pickle.load(f)
    except:
        metadata = {}

    # 🔑 STORE IN A CONSISTENT STRUCTURE
    metadata[filename] = {
    "summary": analysis.get("summary", "") if isinstance(analysis, dict) else analysis
}


    with open(DOC_META_FILE, "wb") as f:
        pickle.dump(metadata, f)

