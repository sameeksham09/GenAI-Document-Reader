from retriver import retrieve_context
from prompts import get_instruction, build_prompt
from generator import generate_answer

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

    context = "\n".join(retrieved_chunks)

    prompt = build_prompt(context, instruction, question)
    answer = generate_answer(prompt)

    print("\n🤖 AI Output:")
    print(answer)
