# logger.py

import json
import os
from datetime import datetime

LOG_FILE = "activity_log.jsonl"


def _write_event(event_dict):
    event_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event_dict) + "\n")


def log_upload(filename, num_chunks):
    """
    Log a document upload event.

    Args:
        filename   : name of the uploaded file e.g. "DBMS.pdf"
        num_chunks : how many chunks were indexed from this file

    Writes:
        {"timestamp": "...", "event": "upload", "filename": "DBMS.pdf", "chunks": 29}
    """
    _write_event({
        "event":    "upload",
        "filename": filename,
        "chunks":   num_chunks,
    })


def log_delete(filename):
    """
    Log a document deletion event.

    Args:
        filename : name of the deleted file

    Writes:
        {"timestamp": "...", "event": "delete", "filename": "resume.pdf"}
    """
    _write_event({
        "event":    "delete",
        "filename": filename,
    })


def log_query(question, rewritten_query, selected_doc, num_chunks_retrieved):
    """
    Log a query event.

    Args:
        question             : the original user question
        rewritten_query      : the rewritten query sent to ChromaDB (may be same as question)
        selected_doc         : which document was searched
        num_chunks_retrieved : how many chunks came back

    Writes:
        {"timestamp": "...", "event": "query", "question": "...",
         "rewritten": "...", "doc": "...", "chunks_retrieved": 4}

    WHY LOG THE REWRITTEN QUERY:
        The rewritten query is what actually hit ChromaDB.
        If retrieval fails, comparing the original question vs rewritten
        query tells you immediately whether the rewriter caused the problem.
    """
    _write_event({
        "event":             "query",
        "question":          question,
        "rewritten":         rewritten_query,
        "doc":               selected_doc,
        "chunks_retrieved":  num_chunks_retrieved,
    })


def load_logs(last_n=50):
    """
    Load the most recent N log entries for display in the UI.

    WHY LAST N ONLY:
        Log files grow indefinitely. Loading all of them into the UI
        would slow down Streamlit on every rerun. We only show the
        last 50 events — enough to see recent activity.

    Returns:
        List of event dicts, most recent LAST (chronological order).
        Empty list if log file doesn't exist yet.
    """
    if not os.path.exists(LOG_FILE):
        return []

    lines = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue   # skip malformed lines

    # Return last N entries (most recent)
    return lines[-last_n:]


def get_stats():
    """
    Compute summary statistics from the log file.

    Returns a dict with:
        total_uploads   : how many documents were ever uploaded
        total_deletes   : how many documents were ever deleted
        total_queries   : how many questions were asked
        active_docs     : uploads minus deletes (currently indexed docs)
        most_queried    : which document was asked about most

    Used to show a stats panel in the UI.
    """
    logs = load_logs(last_n=10000)   # load all for stats

    uploads  = [l for l in logs if l["event"] == "upload"]
    deletes  = [l for l in logs if l["event"] == "delete"]
    queries  = [l for l in logs if l["event"] == "query"]

    # Count queries per document
    doc_query_counts = {}
    for q in queries:
        doc = q.get("doc", "unknown")
        doc_query_counts[doc] = doc_query_counts.get(doc, 0) + 1

    most_queried = max(doc_query_counts, key=doc_query_counts.get) \
                   if doc_query_counts else "—"

    return {
        "total_uploads":  len(uploads),
        "total_deletes":  len(deletes),
        "total_queries":  len(queries),
        "active_docs":    len(uploads) - len(deletes),
        "most_queried":   most_queried,
    }