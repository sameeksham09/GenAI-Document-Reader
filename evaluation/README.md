# RAG evaluation

This folder holds a **held-out test set** and a **runner script** to evaluate your RAG pipeline (retrieval + LLM) with simple metrics.

## Before you run

The default test set expects content from **`notes.txt`**. Use one of these:

**Option A – Clean index (recommended for eval)**  
If you’ve already uploaded other docs and get “First question retrieved 0 chunks”, the index may be out of sync. Reset and index only `notes.txt`:

```bash
python evaluation/reset_index_for_eval.py
```

**Option B – Add notes.txt to the existing index**  
If the index is empty and you only want to add `notes.txt`:

```bash
python rag.py
```

Then run the evaluation (see below). If the index is empty or doesn’t contain `notes.txt`, the script will print a short message and exit.

## Test set

- **`test_qa.json`**: list of `{"question": "...", "reference": "...", "source_doc": "notes.txt"}`.
  - `source_doc` (optional) restricts retrieval to that document.

Add your own examples by appending entries; the references should be short gold answers from the document(s) you index.

## Run evaluation

From the **project root**:

```bash
python evaluation/run_eval.py
```

Optional env vars (same as the main app):

- `LLM_BACKEND=ollama` (default) | `openai` | `local_lora`
- `EVAL_OUTPUT=results.json` — write per-example results and summary to a JSON file
- `EVAL_DEBUG=1` — print chunk count and a short context snippet per question (to debug retrieval)

## Metrics

- **Token F1**: word-overlap F1 between model answer and reference (lowercased). Higher is better.
- **Exact match**: reference and prediction match after stripping and lowercasing.

The script prints average F1, exact-match count, and per-example breakdown.
