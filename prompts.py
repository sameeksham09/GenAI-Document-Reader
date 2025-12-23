def get_instruction(qtype, num_questions=1):
    if qtype == "1":
        return "Answer the question clearly."
    elif qtype == "2":  # MCQ
        return f"Create {num_questions} multiple-choice questions and clearly mark the correct answers."
    elif qtype == "3":  # True/False
        return f"Create {num_questions} True/False questions and provide one-line justification."
    elif qtype == "4":
        return "Create a fill-in-the-blank question and provide the answer."
    else:
        return None


STYLE_GUARD = """
Do not explain concepts beyond the document.
Do not introduce new definitions.
"""


def build_prompt(context, instruction, question):
    return f"""
Use ONLY the context below.
If the answer can be inferred from the context, answer clearly in your own words.
Do NOT introduce information not present in the context.

Context:
{context}

Instruction:
{instruction}

{STYLE_GUARD}

Question:
{question}
"""
