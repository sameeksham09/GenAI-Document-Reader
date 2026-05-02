# debug_chunks.py
# Run from your project root: python debug_chunks.py

import pickle

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Total chunks indexed: {len(chunks)}")
print(f"{'─'*60}")

# Check for duplicate text
from collections import Counter
import hashlib

hashes = [hashlib.md5(c["text"].strip().lower().encode()).hexdigest() for c in chunks]
dupes  = [h for h, count in Counter(hashes).items() if count > 1]
print(f"🔁 Duplicate chunks: {len(dupes)}")

# Check sentence boundaries
bad_endings = 0
for c in chunks:
    if c["text"].strip() and c["text"].strip()[-1] not in ".!?\"'":
        bad_endings += 1

print(f"⚠️  Chunks with abrupt endings: {bad_endings}")
print(f"{'─'*60}")

# Print first 5 chunks
print("\n📄 First 5 chunks:\n")
for c in chunks[:5]:
    print(f"  Chunk {c['id']} | source: {c['source']}")
    print(f"  {c['text'][:200]}...")
    print()

# Print last 2 chunks
print(f"{'─'*60}")
print("\n📄 Last 2 chunks:\n")
for c in chunks[-2:]:
    print(f"  Chunk {c['id']} | source: {c['source']}")
    print(f"  {c['text'][:200]}...")
    print()

# Score distribution hint
print(f"{'─'*60}")
print(f"\n📊 Chunk length stats:")
lengths = [len(c["text"].split()) for c in chunks]
print(f"  Min words : {min(lengths)}")
print(f"  Max words : {max(lengths)}")
print(f"  Avg words : {sum(lengths)//len(lengths)}")