"""One-off helper script to index `notes.txt` using the canonical retriever path."""

from retriever import index_text_document


def main():
    with open("notes.txt", "r") as f:
        text = f.read()

    num_chunks = index_text_document(text, source="notes.txt")

    print("Number of chunks added:", num_chunks)
    print("✅ Document indexed successfully via canonical path.")


if __name__ == "__main__":
    main()
