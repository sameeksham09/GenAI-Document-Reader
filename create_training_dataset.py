import os
import json
from PyPDF2 import PdfReader

from generator import generate_answer
from prompts import build_prompt

# ---------------- CONFIG ----------------
PDF_FOLDER = "./pdf_docs/"
CHUNK_SIZE = 300  # number of words per chunk
OUTPUT_FILE = "training_data.jsonl"
AUTO_ANSWER = True  # Set False if you want empty answers

# ---------------- LOAD PDFs ----------------
pdf_texts = {}

for file in os.listdir(PDF_FOLDER):
    if file.lower().endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)
        reader = PdfReader(path)

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        pdf_texts[file] = text

print(f"Loaded {len(pdf_texts)} PDFs.")

# ---------------- CREATE DATASET ----------------
dataset_entries = []

for pdf_name, text in pdf_texts.items():
    words = text.split()

    for i in range(0, len(words), CHUNK_SIZE):
        chunk_text = " ".join(words[i:i + CHUNK_SIZE])

        entry = {
            "context": chunk_text,
            "question": f"Summarize this chunk from {pdf_name}",
            "answer": ""
        }

        if AUTO_ANSWER:
            try:
                prompt = build_prompt(
                    chunk_text,
                    "Summarize the content",
                    entry["question"]
                )
                entry["answer"] = generate_answer(prompt)
            except Exception as e:
                print("Generation failed:", e)

        dataset_entries.append(entry)

print(f"Created {len(dataset_entries)} dataset entries.")

# ---------------- SAVE DATASET ----------------
with open(OUTPUT_FILE, "w") as f:
    for entry in dataset_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Dataset saved as {OUTPUT_FILE}")
