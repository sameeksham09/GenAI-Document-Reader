from retriver import retrieve_context
from prompts import get_instruction, build_prompt
from generator import generate_answer

DEBUG = True  # 🔥 Phase 5 flag


def normalize_question(q):
    q = q.strip()
    if not q.endswith("?"):
        q += "?"
    return q


print("📄 RAG system ready.")

print("""
Choose question type:
1 - Descriptive
2 - MCQ
3 - True / False
4 - Fill in the blanks
""")

while True:
    qtype = input("Enter question type (1/2/3/4 or exit): ").strip()
    if qtype.lower() == "exit":
        break

    instruction = get_instruction(qtype)
    if not instruction:
        print("Invalid option.")
        continue

    raw_question = input("Enter your question: ")
    question = normalize_question(raw_question)

    retrieved_chunks = retrieve_context(question)

    if not retrieved_chunks:
        print("🤖 AI Answer: I don't know based on the provided document.")
        continue

    # 🔍 Phase 5: Evaluation Output
    if DEBUG:
        print("\n🔍 Evaluation Mode ON")
        print("Retrieved Chunks:")
        for c in retrieved_chunks:
            print(f"- Chunk {c['id']} | score: {round(c['score'], 4)}")

    # Build context and sources
    context = "\n".join([c["text"] for c in retrieved_chunks])
    sources = [f"notes.txt | chunk {c['id']}" for c in retrieved_chunks]

    prompt = build_prompt(context, instruction, question)

    # (Optional – we’ll use this in next step)
    if DEBUG:
        print("\n🧠 Prompt Sent to LLM:")
        print(prompt)

    answer = generate_answer(prompt)

    print("\n🤖 AI Output:")
    print(answer)

    print("\n📌 Sources:")
    for src in sources:
        print("-", src)
