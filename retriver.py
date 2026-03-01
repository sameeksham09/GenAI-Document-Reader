"""Document retrieval and indexing utilities for the RAG system.

This module now owns the *canonical* indexing path used by both:
- CLI utilities (e.g. one-off scripts to index existing files)
- The Streamlit UI (dynamic document upload)
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import io
import os
from PyPDF2 import PdfReader
from doc_analyzer import analyze_document

SIMILARITY_THRESHOLD = 1.8
CHUNK_SIZE = 300

CHUNKS_FILE = "chunks.pkl"
INDEX_FILE = "doc_index.faiss"
DOC_LIST_FILE = "uploaded_docs.pkl"
DOC_META_FILE = "doc_metadata.pkl"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
try:
    index = faiss.read_index(INDEX_FILE)
except Exception:
    # 384 is the dimension of all-MiniLM-L6-v2 embeddings
    index = faiss.IndexFlatL2(384)

# Load chunks
try:
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
except Exception:
    chunks = []


def _save_index_and_chunks():
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, INDEX_FILE)


def _load_doc_list():
    if os.path.exists(DOC_LIST_FILE):
        with open(DOC_LIST_FILE, "rb") as f:
            return pickle.load(f)
    return []


def _save_doc_list(doc_list):
    with open(DOC_LIST_FILE, "wb") as f:
        pickle.dump(doc_list, f)


def _load_metadata():
    if os.path.exists(DOC_META_FILE):
        with open(DOC_META_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_metadata(metadata):
    with open(DOC_META_FILE, "wb") as f:
        pickle.dump(metadata, f)


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Simple word-based chunking used consistently across the app."""
    words = text.split()
    out_chunks = []
    for i in range(0, len(words), chunk_size):
        out_chunks.append(" ".join(words[i : i + chunk_size]))
    return out_chunks


def index_text_document(text, source):
    """Canonical path to index a single text document into FAISS + chunks store."""
    global chunks, index

    text = text.strip()
    if not text:
        return 0

    text_chunks = chunk_text(text)
    if not text_chunks:
        return 0

    start_id = len(chunks)
    new_chunks = []
    for i, ch in enumerate(text_chunks):
        new_chunks.append(
            {
                "id": start_id + i,
                "text": ch,
                "source": source,
            }
        )

    embeddings = embed_model.encode([c["text"] for c in new_chunks])
    embeddings = np.array(embeddings, dtype=np.float32)

    index.add(embeddings)

    chunks.extend(new_chunks)
    _save_index_and_chunks()

    return len(new_chunks)


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
        if selected_doc and chunk["source"] != selected_doc:
            continue
        retrieved.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score),
            }
        )
        if len(retrieved) == k:
            break
    return retrieved


def add_new_document(file_bytes, filename):
    """
    Add a new TXT or PDF document:
    - Extract text
    - Index into FAISS via the canonical indexing path
    - Update doc list and metadata (summary)
    """
    name = filename
    ext = os.path.splitext(name)[1].lower()
    text = ""

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        except Exception as e:
            print(f"Failed to read PDF {name}: {e}")
            return False
    else:
        # Treat everything else as text
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"Failed to decode text file {name}: {e}")
            return False

    if not text.strip():
        print(f"No text content found in {name}")
        return False

    # Update doc list
    doc_list = _load_doc_list()
    if name not in doc_list:
        doc_list.append(name)
        _save_doc_list(doc_list)

    # Index content
    num_chunks = index_text_document(text, source=name)
    if num_chunks == 0:
        return False

    # Generate and store summary metadata (best-effort)
    try:
        summary = analyze_document(text)
        metadata = _load_metadata()
        metadata[name] = {"summary": summary}
        _save_metadata(metadata)
    except Exception as e:
        print(f"Failed to generate summary for {name}: {e}")

    return True
