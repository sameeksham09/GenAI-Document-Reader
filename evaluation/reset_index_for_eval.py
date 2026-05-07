"""
Reset the FAISS index and chunks to contain only notes.txt.
Use this when the index is out of sync (e.g. more chunks than vectors)
or when you want a clean index for evaluation.

Run from project root:
  python evaluation/reset_index_for_eval.py

Then run evaluation:
  python evaluation/run_eval.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Remove index files so retriver loads empty state on import
for name in ("chunks.pkl", "doc_index.faiss"):
    path = os.path.join(PROJECT_ROOT, name)
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed {name}")
    else:
        print(f"Not found (ok): {name}")

# Now import: retriver will see missing files and start with empty index/chunks
from retriever import index_text_document

notes_path = os.path.join(PROJECT_ROOT, "notes.txt")
if not os.path.exists(notes_path):
    print("notes.txt not found. Create it in the project root and run again.")
    sys.exit(1)

with open(notes_path, "r") as f:
    text = f.read()

n = index_text_document(text, source="notes.txt")
print(f"Indexed notes.txt: {n} chunk(s).")
print("You can now run: python evaluation/run_eval.py")
