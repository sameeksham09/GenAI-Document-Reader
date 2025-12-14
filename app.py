import ollama

# Read the document
with open("notes.txt", "r") as f:
    document = f.read()

print("📄 Document loaded. Ask questions based on it.")

while True:
    question = input("\nAsk a question (or type exit): ")
    if question.lower() == "exit":
        print("Goodbye!")
        break

    prompt = f"""
    You are an assistant that answers ONLY using the document below.

    Document:
    {document}

    Question:
    {question}
    """

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n🤖 AI Answer:")
    print(response["message"]["content"])
