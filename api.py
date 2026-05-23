# api.py
"""
FastAPI REST layer for the RAG system.

WHY THIS EXISTS:
─────────────────────────────────────────────────────────────────────────────
The Streamlit UI works for demos but is browser-only.
This API wraps the same RAG pipeline in HTTP endpoints so any application
can use it — mobile apps, web frontends, other microservices, scripts.

This is the standard production pattern:
  - Streamlit = internal demo / prototyping interface
  - FastAPI   = production API that external systems call

Both use the EXACT same backend (retriever.py, llm_utils.py, prompts.py).
Only the interface layer is different.

HOW TO RUN:
    uvicorn api:app --reload --port 8000

HOW TO TEST (interactive docs):
    http://localhost:8000/docs     ← Swagger UI, test every endpoint in browser
    http://localhost:8000/redoc   ← Alternative docs view

ENDPOINTS:
    GET    /health                → system status
    GET    /documents             → list indexed documents
    POST   /ingest                → upload and index a document
    POST   /query                 → ask a question, get answer + faithfulness
    DELETE /document/{filename}   → remove a document from index
─────────────────────────────────────────────────────────────────────────────
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

from retriever import (
    retrieve_context,
    add_new_document,
    get_all_documents,
    delete_document,
)
from prompts import get_instruction, build_prompt, build_rewrite_prompt
from llm_utils import generate_answer, check_faithfulness

# ── App setup ──────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS FastAPI():
#   Creates the application instance.
#   title, description, version appear in the auto-generated /docs page.
#   This is what uvicorn serves when you run: uvicorn api:app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG QA System API",
    description="""
Production REST API for the Retrieval-Augmented Generation document QA system.

## Features
- **Hybrid retrieval**: BM25 + dense vector search with RRF fusion
- **Cross-encoder reranking**: ms-marco-MiniLM-L-6-v2
- **Conversation memory**: multi-turn context injection
- **Faithfulness scoring**: LLM-as-judge answer verification
- **Streaming**: available via Streamlit UI

## Quick Start
1. Upload a document via `POST /ingest`
2. Ask questions via `POST /query`
3. Manage documents via `GET /documents` and `DELETE /document/{filename}`
    """,
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS MIDDLEWARE:
#   CORS = Cross-Origin Resource Sharing.
#   Without this, a web browser blocks requests from a different domain.
#   Example: your React frontend at localhost:3000 calling this API at
#   localhost:8000 would be blocked by the browser's same-origin policy.
#   allow_origins=["*"] allows ALL origins — fine for development.
#   In production, replace "*" with your specific frontend domain.
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, DELETE, etc.
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# WHAT ARE PYDANTIC MODELS:
#   Pydantic models define the shape of request bodies and responses.
#   FastAPI uses them to:
#   1. Validate incoming JSON automatically (wrong types = 422 error)
#   2. Generate accurate API documentation at /docs
#   3. Serialize Python objects to JSON for responses
#
#   BaseModel = Pydantic's base class for data models.
#   Every field has a type annotation. Optional fields have defaults.
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    question:      str                    # the user's question (required)
    selected_doc:  Optional[str] = None   # which document to search (optional)
    chat_history:  Optional[list] = None  # previous turns for memory (optional)
    check_faithful: Optional[bool] = True # whether to run faithfulness check


class QueryResponse(BaseModel):
    """Response body for POST /query"""
    question:          str
    rewritten_query:   str
    answer:            str
    sources:           list              # list of retrieved chunk dicts
    faithfulness:      Optional[dict]   # faithfulness score dict or None
    latency_ms:        float            # total time in milliseconds


class IngestResponse(BaseModel):
    """Response body for POST /ingest"""
    filename:    str
    chunks_added: int
    message:     str


class DocumentsResponse(BaseModel):
    """Response body for GET /documents"""
    documents: list[str]
    count:     int


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status:  str
    version: str
    models:  dict


class DeleteResponse(BaseModel):
    """Response body for DELETE /document/{filename}"""
    filename: str
    message:  str


# ── Endpoints ──────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: GET /health
#
# WHY THIS EXISTS:
#   Every production API has a health endpoint.
#   It's used by:
#   - Load balancers to check if the service is up
#   - Monitoring tools (Datadog, Grafana) to alert on downtime
#   - Docker/Kubernetes to know when to restart the container
#   - You, to quickly verify the API started correctly
#
# Returns system status and which models are loaded.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API is running and all models are loaded."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models={
            "embedder":  "all-MiniLM-L6-v2",
            "reranker":  "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm":       "llama3.2 (via Ollama)",
            "vector_db": "ChromaDB (persistent)",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: GET /documents
#
# Lists all currently indexed documents by querying ChromaDB metadata.
# Same as what the Streamlit dropdown shows.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
def list_documents():
    """List all indexed documents available for querying."""
    docs = get_all_documents()
    return DocumentsResponse(documents=docs, count=len(docs))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: POST /ingest
#
# WHAT IT DOES:
#   Accepts a file upload, runs the full ingestion pipeline:
#   extract text → clean → chunk → embed → store in ChromaDB + BM25
#
# HOW FILE UPLOAD WORKS IN FASTAPI:
#   UploadFile is FastAPI's type for multipart file uploads.
#   file.read() gives us the raw bytes — same as what Streamlit passes.
#   So we can call add_new_document() directly with the same interface.
#
# HTTP STATUS CODES:
#   200 → success
#   400 → bad request (wrong file type, empty file)
#   409 → conflict (document already indexed)
#   500 → server error
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF or TXT document.

    The document will be:
    - Extracted and cleaned (PDF artifacts removed)
    - Chunked into sentence-aware windows with overlap
    - Embedded and stored in ChromaDB
    - Tokenized and stored in BM25 index
    """
    # Validate file type
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Only PDF and TXT are supported."
        )

    # Read file bytes
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Check if already indexed
    existing_docs = get_all_documents()
    if file.filename in existing_docs:
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' is already indexed. Delete it first to re-index."
        )

    # Run ingestion pipeline
    success = add_new_document(file_bytes, file.filename)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to index document. Check server logs for details."
        )

    # Get chunk count for response
    from retriever import collection
    result   = collection.get(where={"source": file.filename})
    n_chunks = len(result["ids"]) if result["ids"] else 0

    return IngestResponse(
        filename=file.filename,
        chunks_added=n_chunks,
        message=f"Successfully indexed '{file.filename}' into {n_chunks} chunks."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: POST /query  (the main endpoint)
#
# WHAT IT DOES:
#   Runs the complete RAG pipeline and returns a structured JSON response.
#
# PIPELINE (same as Streamlit UI):
#   1. Query rewriting   (if chat_history provided)
#   2. Hybrid retrieval  (BM25 + ChromaDB → RRF → cross-encoder rerank)
#   3. LLM generation    (with conversation memory)
#   4. Faithfulness check (optional, adds ~2-3s)
#
# LATENCY TRACKING:
#   We record start time and compute total latency in milliseconds.
#   This is returned in the response so callers can monitor performance.
#   In production this would be logged to a metrics system.
#
# WHY NOT STREAMING HERE:
#   REST APIs return complete responses — you can't stream JSON.
#   Streaming is a UI concern (Streamlit handles it).
#   The API returns the complete answer once generation is done.
#   If streaming is needed, a WebSocket endpoint would be the right approach.
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_document(request: QueryRequest):
    """
    Ask a question and get an answer from indexed documents.

    Runs the full pipeline:
    - Query rewriting for follow-up questions
    - Hybrid BM25 + dense retrieval with RRF fusion
    - Cross-encoder reranking
    - LLM answer generation with conversation memory
    - Optional faithfulness scoring
    """
    start_time = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chat_history = request.chat_history or []

    # ── Step 1: Query rewriting ───────────────────────────────────────────────
    # If conversation history is provided, rewrite the question to resolve
    # pronouns and references before sending to retrieval.
    if chat_history:
        rewrite_prompt  = build_rewrite_prompt(request.question, chat_history)
        rewritten_query = generate_answer(rewrite_prompt, max_new_tokens=80)

        # Extract first line only (small LLMs tend to ramble)
        lines = [l.strip() for l in rewritten_query.splitlines() if l.strip()]
        rewritten_query = lines[0] if lines else request.question

        # Strip label prefixes
        for prefix in ["Rewritten query:", "Rewritten:", "Query:", "Question:"]:
            if rewritten_query.lower().startswith(prefix.lower()):
                rewritten_query = rewritten_query[len(prefix):].strip()
        rewritten_query = rewritten_query.strip('"\'')

        if len(rewritten_query) < 5:
            rewritten_query = request.question
    else:
        rewritten_query = request.question

    # ── Step 2: Hybrid retrieval ──────────────────────────────────────────────
    retrieved = retrieve_context(
        rewritten_query,
        k=4,
        selected_doc=request.selected_doc
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the indexed documents for this question."
        )

    # ── Step 3: Build prompt + generate answer ────────────────────────────────
    context     = "\n".join(c["text"] for c in retrieved)
    instruction = get_instruction("1")   # descriptive mode
    prompt      = build_prompt(
        context,
        instruction,
        request.question,
        chat_history=chat_history
    )
    answer = generate_answer(prompt, max_new_tokens=300)

    # ── Step 4: Faithfulness check (optional) ────────────────────────────────
    faithfulness = None
    if request.check_faithful:
        faithfulness = check_faithfulness(context, answer)

    # ── Step 5: Build response ────────────────────────────────────────────────
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Clean up sources for JSON serialization
    sources = [
        {
            "text":         c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            "source":       c["source"],
            "rrf_score":    round(c.get("score", 0), 4),
            "rerank_score": round(c.get("rerank_score", 0), 4),
        }
        for c in retrieved
    ]

    return QueryResponse(
        question=request.question,
        rewritten_query=rewritten_query,
        answer=answer,
        sources=sources,
        faithfulness=faithfulness,
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5: DELETE /document/{filename}
#
# WHAT IT DOES:
#   Removes a document from both ChromaDB and BM25 index.
#   This is the surgical delete that was impossible with FAISS.
#
# PATH PARAMETER:
#   {filename} in the URL is a path parameter.
#   DELETE /document/DBMS.pdf → filename = "DBMS.pdf"
#   FastAPI automatically extracts it and passes to the function.
# ─────────────────────────────────────────────────────────────────────────────

@app.delete("/document/{filename}", response_model=DeleteResponse, tags=["Documents"])
def remove_document(filename: str):
    """
    Remove a document from the index.

    Deletes all chunks belonging to this document from:
    - ChromaDB vector store
    - BM25 keyword index
    - Document metadata (summaries)
    """
    existing_docs = get_all_documents()
    if filename not in existing_docs:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in index."
        )

    success = delete_document(filename)

    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete '{filename}'."
        )

    return DeleteResponse(
        filename=filename,
        message=f"'{filename}' successfully removed from index."
    )