"""Document retrieval and indexing utilities for the RAG system."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
import io
import os
import re
import hashlib
from PyPDF2 import PdfReader
from doc_analyzer import analyze_document

# ── Constants ──────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 2.5
CHUNK_SIZE           = 5
CHUNK_OVERLAP        = 2

CHUNKS_FILE   = "chunks.pkl"
INDEX_FILE    = "doc_index.faiss"
DOC_LIST_FILE = "uploaded_docs.pkl"
DOC_META_FILE = "doc_metadata.pkl"

# ── Models ─────────────────────────────────────────────────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Load persisted state ───────────────────────────────────────────────────────
try:
    index = faiss.read_index(INDEX_FILE)
except Exception:
    index = faiss.IndexFlatL2(384)

try:
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
except Exception:
    chunks = []


# ── Persistence helpers ────────────────────────────────────────────────────────

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

def _text_hash(text):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def _get_existing_hashes():
    return {_text_hash(c["text"]) for c in chunks}


# ── PDF cleaning ───────────────────────────────────────────────────────────────
#
# WHAT WAS WRONG:
#   The debug output showed chunks like:
#     "Database \nManagement \nSystems (DBMS) \n1. Introduction to \nDBMS \nA \nDatabase..."
#
#   This is PyPDF2 extracting styled/formatted PDF headings character by character.
#   Each word in a bold heading becomes its own line. The previous cleaner only
#   handled single-newline mid-sentence breaks, but not this "every word on its
#   own line" pattern that comes from PDF font/layout extraction artifacts.
#
# ROOT CAUSE OF NEGATIVE RERANK SCORES:
#   When a chunk contains "Database \nManagement \nSystems" instead of
#   "Database Management Systems", the sentence embedder sees fragmented tokens.
#   The embedding drifts far from what "ACID properties" or "transactions" would
#   produce — so FAISS retrieves marginally related chunks, and the cross-encoder
#   (which reads the raw text) correctly scores them very low (negative = bad match).
#
# FIX — _clean_pdf_text() is now more aggressive:
#   Step 1: Collapse "word \n word \n word" patterns — any newline where the
#           next line starts with a lowercase letter OR is just a short word
#           (< 4 chars) gets replaced with a space. This handles styled headings.
#   Step 2: Collapse hyphenated line breaks ("transac-\ntion" → "transaction").
#   Step 3: Remove standalone page numbers.
#   Step 4: Normalise remaining whitespace.
#   Step 5: Join pages with "\n\n" so paragraph structure is preserved.

def _clean_pdf_text(raw_pages):
    """
    Aggressively clean PyPDF2 page text and join into one structured string.

    Handles:
    - Styled heading extraction ("Database \\nManagement \\nSystems" → one line)
    - Hyphenated line breaks
    - Standalone page numbers
    - Excessive whitespace
    """
    cleaned = []
    for page in raw_pages:
        if not page:
            continue

        # Step 1: Remove standalone page-number lines (just digits, optional spaces)
        page = re.sub(r'^\s*\d+\s*$', '', page, flags=re.MULTILINE)

        # Step 2: Collapse hyphenated line breaks ("transac-\ntion" → "transaction")
        page = re.sub(r'-\n', '', page)

        # Step 3: Collapse newlines where the NEXT line is short (≤3 chars) or
        # starts with lowercase — these are mid-word / mid-heading line breaks
        # from PyPDF2's character-by-character styled-text extraction.
        # Pattern: \n followed by a short token or lowercase continuation.
        page = re.sub(r'\n(?=[a-z])', ' ', page)          # lowercase continuation
        page = re.sub(r'\n(?=\S{1,3}\s)', ' ', page)      # very short next word

        # Step 4: Any remaining single newline that isn't starting a real new
        # sentence (uppercase after a period is a real sentence — keep those).
        # Replace single \n not preceded by sentence-ending punctuation with space.
        page = re.sub(r'(?<![.!?])\n(?!\n)', ' ', page)

        # Step 5: Collapse multiple spaces
        page = re.sub(r' {2,}', ' ', page)

        page = page.strip()
        if page:
            cleaned.append(page)

    return "\n\n".join(cleaned)


# ── Sentence splitting ─────────────────────────────────────────────────────────

def _split_into_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = []
    for fragment in raw:
        for part in re.split(r'\n{2,}', fragment):
            part = part.strip()
            if len(part) > 20:
                sentences.append(part)
    return sentences


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sentences = _split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= chunk_size:
        return [" ".join(sentences)]
    step = max(1, chunk_size - overlap)
    out, i = [], 0
    while i < len(sentences):
        out.append(" ".join(sentences[i : i + chunk_size]))
        i += step
    return out


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_text_document(text, source):
    global chunks, index

    text = text.strip()
    if not text:
        return 0

    text_chunks     = chunk_text(text)
    existing_hashes = _get_existing_hashes()
    new_chunks      = []

    for ch in text_chunks:
        h = _text_hash(ch)
        if h in existing_hashes:
            continue
        existing_hashes.add(h)
        new_chunks.append({
            "id":     len(chunks) + len(new_chunks),
            "text":   ch,
            "source": source,
        })

    if not new_chunks:
        return 0

    embeddings = embed_model.encode([c["text"] for c in new_chunks])
    index.add(np.array(embeddings, dtype=np.float32))
    chunks.extend(new_chunks)
    _save_index_and_chunks()
    return len(new_chunks)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_context(question, k=4, selected_doc=None):
    question_emb = np.array(embed_model.encode([question]), dtype=np.float32)

    if index.ntotal == 0:
        return []

    D, I        = index.search(question_emb, k * 3)
    candidates  = []
    seen_hashes = set()

    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        if score > SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[idx]
        th    = _text_hash(chunk["text"])
        if th in seen_hashes:
            continue
        seen_hashes.add(th)

        if selected_doc and chunk.get("source") != selected_doc:
            continue

        candidates.append({
            "id":     chunk["id"],
            "text":   chunk["text"],
            "source": chunk.get("source", ""),
            "score":  float(score),
        })

    if not candidates:
        return []

    pairs         = [(question, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:k]


# ── Document ingestion ─────────────────────────────────────────────────────────

def add_new_document(file_bytes, filename):
    name = filename
    ext  = os.path.splitext(name)[1].lower()
    text = ""

    if ext == ".pdf":
        try:
            reader    = PdfReader(io.BytesIO(file_bytes))
            raw_pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            text      = _clean_pdf_text(raw_pages)
        except Exception as e:
            print(f"Failed to read PDF {name}: {e}")
            return False
    else:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"Failed to decode {name}: {e}")
            return False

    if not text.strip():
        print(f"No text found in {name}")
        return False

    doc_list = _load_doc_list()
    if name not in doc_list:
        doc_list.append(name)
        _save_doc_list(doc_list)

    num_chunks = index_text_document(text, source=name)
    if num_chunks == 0:
        return False

    try:
        summary  = analyze_document(text[:3000])
        metadata = _load_metadata()
        metadata[name] = {"summary": summary}
        _save_metadata(metadata)
    except Exception as e:
        print(f"Summary failed for {name}: {e}")

    return True