"""
retriever.py — Hybrid retrieval: BM25 + ChromaDB dense search via RRF fusion.

WHAT CHANGED FROM CHROMADB-ONLY VERSION:
─────────────────────────────────────────────────────────────────────────────
BEFORE (ChromaDB only):
    retrieve_context() → ChromaDB dense search → cross-encoder rerank → top k
    Only used semantic (vector) similarity.
    Struggled with exact-match queries: "What is BCNF?", "What does DML stand for?"

AFTER (Hybrid BM25 + Dense):
    retrieve_context() → BM25 keyword search  ┐
                       → ChromaDB dense search ┼→ RRF fusion → cross-encoder rerank → top k
    Combines keyword precision with semantic recall.
    BM25 catches exact terms. Dense catches synonyms and meaning.
    RRF merges both ranked lists into one superior ranking.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import io
import re
import pickle

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader
from doc_analyzer import analyze_document
from logger import log_upload, log_delete

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 5
CHUNK_OVERLAP = 2
TOP_K_FETCH   = 12    # candidates fetched from EACH retriever before fusion
RRF_K         = 60    # RRF constant — standard value from IR literature

CHROMA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DOC_META_FILE = "doc_metadata.pkl"
BM25_FILE     = "bm25_index.pkl"   # NEW: persisted BM25 index

# ── Models ─────────────────────────────────────────────────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── ChromaDB ───────────────────────────────────────────────────────────────────
client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# ── BM25 Index ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# WHY BM25 NEEDS ITS OWN STORAGE:
#
# ChromaDB stores embeddings (vectors of 384 floats).
# BM25 works on raw token lists — it needs the actual words.
# ChromaDB doesn't expose the token lists it uses internally.
# So we maintain a parallel lightweight store: bm25_index.pkl
#
# It contains:
#   bm25_corpus : list of tokenized chunks  [["what", "is", "dbms"], [...], ...]
#   bm25_docs   : list of raw chunk texts   ["What is a DBMS?...", ...]
#   bm25_metas  : list of metadata dicts    [{"source": "DBMS.pdf"}, ...]
#
# This mirrors exactly what's in ChromaDB, kept in sync on every add/delete.
# It's small — text tokens, not vectors — so the pkl file stays tiny.
#
# BM25Okapi is created fresh from the corpus on each app start (fast).
# ─────────────────────────────────────────────────────────────────────────────

def _load_bm25_store():
    """Load the BM25 corpus, docs, and metas from disk."""
    if os.path.exists(BM25_FILE):
        with open(BM25_FILE, "rb") as f:
            return pickle.load(f)
    return {"corpus": [], "docs": [], "metas": []}

def _save_bm25_store(store):
    """Save the BM25 corpus, docs, and metas to disk."""
    with open(BM25_FILE, "wb") as f:
        pickle.dump(store, f)

def _tokenize(text):
    """
    Tokenize text for BM25.

    BM25 works on a list of tokens (words).
    We lowercase and split on non-alphanumeric characters.
    Simple but effective for English technical text.

    Example:
        "What is a DBMS?" → ["what", "is", "a", "dbms"]
    """
    return re.findall(r'\b\w+\b', text.lower())

# Load BM25 store at startup
_bm25_store = _load_bm25_store()

def _build_bm25(corpus):
    """
    Build a BM25Okapi index from a list of token lists.

    BM25Okapi is the most widely used BM25 variant.
    'Okapi' refers to the Okapi BM25 weighting scheme from
    the Okapi information retrieval system (City University London, 1994).

    If corpus is empty, return None — no documents indexed yet.
    """
    if not corpus:
        return None
    return BM25Okapi(corpus)


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _load_metadata():
    if os.path.exists(DOC_META_FILE):
        with open(DOC_META_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def _save_metadata(metadata):
    with open(DOC_META_FILE, "wb") as f:
        pickle.dump(metadata, f)


# ── Chunk ID ───────────────────────────────────────────────────────────────────

def _make_chunk_id(source, chunk_index):
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", source)
    return f"{clean_name}_chunk_{chunk_index:04d}"


# ── PDF Cleaning ───────────────────────────────────────────────────────────────

def _clean_pdf_text(raw_pages):
    """Join PDF pages into clean prose, collapsing PyPDF2 extraction artifacts."""
    cleaned = []
    for page in raw_pages:
        if not page:
            continue
        page = re.sub(r'^\s*\d+\s*$', '', page, flags=re.MULTILINE)
        page = re.sub(r'-\n', '', page)
        page = re.sub(r'\n(?=[a-z])', ' ', page)
        page = re.sub(r'\n(?=\S{1,3}\s)', ' ', page)
        page = re.sub(r'(?<![.!?])\n(?!\n)', ' ', page)
        page = re.sub(r' {2,}', ' ', page)
        page = page.strip()
        if page:
            cleaned.append(page)
    return "\n\n".join(cleaned)


# ── Sentence splitting + Chunking ──────────────────────────────────────────────

def _split_into_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = []
    for fragment in raw:
        for part in re.split(r'\n{2,}', fragment):
            part = part.strip()
            if len(part) > 20:
                sentences.append(part)
    return sentences

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sentences = _split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= chunk_size:
        return [" ".join(sentences)]
    step = max(1, chunk_size - overlap)
    out, i = [], 0
    while i < len(sentences):
        out.append(" ".join(sentences[i: i + chunk_size]))
        i += step
    return out


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_text_document(text, source):
    """
    Index a document into BOTH ChromaDB (dense) and BM25 (sparse).

    WHAT CHANGED:
    ─────────────────────────────────────────────────────────────────────────
    BEFORE: added chunks only to ChromaDB.
    NOW:    adds chunks to ChromaDB AND appends tokenized chunks to bm25_store.

    Both stores are kept perfectly in sync:
    - Same chunks, same order
    - BM25 store keyed by chunk text so we can match results back
    ─────────────────────────────────────────────────────────────────────────
    """
    global _bm25_store

    text = text.strip()
    if not text:
        return 0

    text_chunks = chunk_text(text)
    if not text_chunks:
        return 0

    # Check which chunks are already in ChromaDB
    existing     = collection.get(where={"source": source})
    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    # Also check which chunks are already in BM25 store
    existing_bm25_texts = set(_bm25_store["docs"])

    new_texts     = []
    new_ids       = []
    new_metadatas = []

    for i, ch in enumerate(text_chunks):
        chunk_id = _make_chunk_id(source, i)
        if chunk_id in existing_ids:
            continue
        new_texts.append(ch)
        new_ids.append(chunk_id)
        new_metadatas.append({"source": source})

    if not new_texts:
        return 0

    # ── Add to ChromaDB (dense vectors) ──────────────────────────────────────
    embeddings = embed_model.encode(new_texts, show_progress_bar=False)
    collection.add(
        documents=new_texts,
        embeddings=[e.tolist() for e in embeddings],
        ids=new_ids,
        metadatas=new_metadatas,
    )

    # ── Add to BM25 store (sparse tokens) ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # WHY WE ADD TO BM25 SEPARATELY:
    # ChromaDB stores vectors, not token lists.
    # BM25 needs token lists to compute term frequency.
    # We store the tokenized version of each chunk alongside the raw text
    # so BM25 can score them and we can retrieve the raw text from the match.
    # ─────────────────────────────────────────────────────────────────────────
    for ch, meta in zip(new_texts, new_metadatas):
        if ch not in existing_bm25_texts:   # avoid duplicates in BM25 too
            _bm25_store["corpus"].append(_tokenize(ch))
            _bm25_store["docs"].append(ch)
            _bm25_store["metas"].append(meta)

    _save_bm25_store(_bm25_store)
    return len(new_texts)


# ── RRF Fusion ─────────────────────────────────────────────────────────────────

def _rrf_fusion(dense_hits, bm25_hits, k=RRF_K):
    """
    Reciprocal Rank Fusion — merge two ranked lists into one.

    HOW RRF WORKS:
    ─────────────────────────────────────────────────────────────────────────
    Each chunk gets a score from EACH retriever based on its rank position:

        rrf_score = 1 / (rank + k)

    where:
        rank = position in the ranked list (1-indexed)
        k    = smoothing constant (60 is standard, prevents top ranks dominating)

    Final score = sum of RRF scores across ALL retrievers.

    EXAMPLE with k=60:
        Chunk A: rank 1 in dense, rank 3 in BM25
            score = 1/(1+60) + 1/(3+60) = 0.0164 + 0.0159 = 0.0323

        Chunk B: rank 2 in dense, rank 1 in BM25
            score = 1/(2+60) + 1/(1+60) = 0.0161 + 0.0164 = 0.0325

        Chunk C: rank 1 in dense only (not in BM25)
            score = 1/(1+60) + 0        = 0.0164

        Chunk B wins even though neither retriever ranked it #1 alone.
        This is the power of fusion — it finds consensus between retrievers.

    WHY k=60:
        Too small (k=1): rank-1 results dominate, fusion barely helps
        Too large (k=100+): all positions treated equally, fusion too flat
        k=60: sweet spot from the original RRF paper (Cormack et al., 2009)
    ─────────────────────────────────────────────────────────────────────────

    Args:
        dense_hits : list of {"text", "source", "score"} from ChromaDB
        bm25_hits  : list of {"text", "source", "score"} from BM25
        k          : RRF smoothing constant

    Returns:
        List of {"text", "source", "rrf_score"} sorted by rrf_score descending
    """
    scores = {}   # text → cumulative RRF score
    metas  = {}   # text → source metadata

    # Score dense results
    for rank, hit in enumerate(dense_hits, start=1):
        text = hit["text"]
        scores[text] = scores.get(text, 0) + 1 / (rank + k)
        metas[text]  = hit["source"]

    # Score BM25 results — add to existing scores (chunks in both get higher total)
    for rank, hit in enumerate(bm25_hits, start=1):
        text = hit["text"]
        scores[text] = scores.get(text, 0) + 1 / (rank + k)
        metas[text]  = hit.get("source", metas.get(text, ""))

    # Sort by RRF score descending
    fused = [
        {"text": text, "source": metas[text], "rrf_score": score}
        for text, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused


# ── BM25 Search ────────────────────────────────────────────────────────────────

def _bm25_search(question, selected_doc=None, top_k=TOP_K_FETCH):
    """
    Search using BM25 keyword matching.

    HOW BM25 SEARCH WORKS:
    ─────────────────────────────────────────────────────────────────────────
    1. Tokenize the question: "What is BCNF?" → ["what", "is", "bcnf"]
    2. BM25Okapi.get_scores() computes a relevance score for every chunk
       based on term frequency and inverse document frequency.
    3. Sort chunks by score descending, return top_k.

    If selected_doc is specified, filter to only chunks from that document
    before running BM25. We rebuild a temporary BM25 index from just those
    chunks — this is fast because our chunks are small in number.
    ─────────────────────────────────────────────────────────────────────────
    """
    global _bm25_store

    if not _bm25_store["corpus"]:
        return []

    corpus = _bm25_store["corpus"]
    docs   = _bm25_store["docs"]
    metas  = _bm25_store["metas"]

    # Filter to selected document if specified
    if selected_doc:
        filtered = [
            (c, d, m) for c, d, m in zip(corpus, docs, metas)
            if m.get("source") == selected_doc
        ]
        if not filtered:
            return []
        corpus, docs, metas = zip(*filtered)
        corpus = list(corpus)
        docs   = list(docs)
        metas  = list(metas)

    # Build BM25 index from (filtered) corpus
    bm25 = _build_bm25(corpus)
    if bm25 is None:
        return []

    # Score all chunks against the question tokens
    query_tokens = _tokenize(question)
    scores       = bm25.get_scores(query_tokens)

    # Sort by score descending, return top_k
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        if scores[idx] > 0:   # only include chunks with at least some match
            results.append({
                "text":   docs[idx],
                "source": metas[idx].get("source", ""),
                "score":  float(scores[idx]),
            })

    return results


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_context(question, k=4, selected_doc=None):
    """
    Hybrid retrieval: BM25 + ChromaDB dense search → RRF fusion → cross-encoder rerank.

    FULL PIPELINE:
    ─────────────────────────────────────────────────────────────────────────
    Step 1 — Dense search (ChromaDB):
        Embeds the question, finds TOP_K_FETCH=12 semantically similar chunks.
        Good for: conceptual questions, synonyms, paraphrasing.
        Weak for: exact rare terms (BCNF, DML abbreviations).

    Step 2 — Sparse search (BM25):
        Tokenizes the question, finds TOP_K_FETCH=12 keyword-matching chunks.
        Good for: exact terms, abbreviations, proper nouns.
        Weak for: semantic meaning, synonyms.

    Step 3 — RRF Fusion:
        Merges the two ranked lists using Reciprocal Rank Fusion.
        Chunks appearing in BOTH lists get higher combined scores.
        Returns a single unified ranked list.

    Step 4 — Cross-encoder reranking:
        Takes top candidates from fused list.
        Cross-encoder reads (question, chunk) together for true relevance score.
        Returns final top k sorted by rerank score.

    WHY THIS ORDER:
        Dense + BM25 are both fast → run both, get 24 total candidates
        RRF reduces to best ~12 → cross-encoder reranks those 12
        Cross-encoder is slow but accurate → only runs on 12 not 24
    ─────────────────────────────────────────────────────────────────────────
    """
    if collection.count() == 0:
        return []

    # ── Step 1: Dense search (ChromaDB) ──────────────────────────────────────
    question_emb = embed_model.encode([question], show_progress_bar=False)
    query_kwargs = {
        "query_embeddings": [question_emb[0].tolist()],
        "n_results":        min(TOP_K_FETCH, collection.count()),
        "include":          ["documents", "metadatas", "distances"],
    }
    if selected_doc:
        query_kwargs["where"] = {"source": selected_doc}

    chroma_results = collection.query(**query_kwargs)
    dense_hits = [
        {
            "text":   doc,
            "source": meta.get("source", ""),
            "score":  float(dist),
        }
        for doc, dist, meta in zip(
            chroma_results["documents"][0],
            chroma_results["distances"][0],
            chroma_results["metadatas"][0],
        )
    ]

    # ── Step 2: BM25 sparse search ────────────────────────────────────────────
    bm25_hits = _bm25_search(question, selected_doc=selected_doc, top_k=TOP_K_FETCH)

    # ── Step 3: RRF Fusion ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # If BM25 returns nothing (all scores 0 — no keyword overlap at all),
    # fall back to dense-only results. This handles very abstract questions
    # where there's no keyword overlap with any chunk.
    # ─────────────────────────────────────────────────────────────────────────
    if bm25_hits:
        fused = _rrf_fusion(dense_hits, bm25_hits)
    else:
        fused = [{"text": h["text"], "source": h["source"], "rrf_score": 0}
                 for h in dense_hits]

    if not fused:
        return []

    # Take top candidates for reranking
    candidates = fused[:TOP_K_FETCH]

    # ── Step 4: Cross-encoder reranking ──────────────────────────────────────
    pairs         = [(question, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return top k with all scores for UI display
    return [
        {
            "text":        c["text"],
            "source":      c["source"],
            "score":       c.get("rrf_score", 0),    # RRF fusion score
            "rerank_score": c["rerank_score"],        # cross-encoder score
        }
        for c in candidates[:k]
    ]


# ── Document management ────────────────────────────────────────────────────────

def get_all_documents():
    """Return list of all indexed document names from ChromaDB metadata."""
    if collection.count() == 0:
        return []
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted(set(m["source"] for m in all_metas if m.get("source")))


def delete_document(filename):
    """
    Delete a document from BOTH ChromaDB and BM25 store.

    WHAT CHANGED:
    ─────────────────────────────────────────────────────────────────────────
    BEFORE: deleted from ChromaDB only.
    NOW:    deletes from ChromaDB AND removes matching entries from bm25_store.

    Both stores must stay in sync — if a chunk is gone from ChromaDB
    but still in BM25, BM25 would return it but ChromaDB couldn't rerank it.
    ─────────────────────────────────────────────────────────────────────────
    """
    global _bm25_store

    # Delete from ChromaDB
    collection.delete(where={"source": filename})

    # Delete from BM25 store
    keep = [
        (c, d, m) for c, d, m in zip(
            _bm25_store["corpus"],
            _bm25_store["docs"],
            _bm25_store["metas"]
        )
        if m.get("source") != filename
    ]

    if keep:
        corpus, docs, metas = zip(*keep)
        _bm25_store = {
            "corpus": list(corpus),
            "docs":   list(docs),
            "metas":  list(metas),
        }
    else:
        _bm25_store = {"corpus": [], "docs": [], "metas": []}

    _save_bm25_store(_bm25_store)

    # Log and remove metadata
    log_delete(filename)
    metadata = _load_metadata()
    if filename in metadata:
        del metadata[filename]
        _save_metadata(metadata)

    return True


def add_new_document(file_bytes, filename):
    """Ingest a new TXT or PDF document into ChromaDB + BM25."""
    ext  = os.path.splitext(filename)[1].lower()
    text = ""

    if ext == ".pdf":
        try:
            reader    = PdfReader(io.BytesIO(file_bytes))
            raw_pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            text      = _clean_pdf_text(raw_pages)
        except Exception as e:
            print(f"Failed to read PDF {filename}: {e}")
            return False
    else:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"Failed to decode {filename}: {e}")
            return False

    if not text.strip():
        print(f"No text found in {filename}")
        return False

    num_chunks = index_text_document(text, source=filename)
    if num_chunks == 0:
        return False

    log_upload(filename, num_chunks)

    try:
        summary  = analyze_document(text[:3000])
        metadata = _load_metadata()
        metadata[filename] = {"summary": summary}
        _save_metadata(metadata)
    except Exception as e:
        print(f"Summary failed for {filename}: {e}")

    return True