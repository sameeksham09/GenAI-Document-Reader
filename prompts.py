def get_instruction(qtype):
    if qtype == "1":
        return "Answer the question clearly."
    elif qtype == "2":
        return "Create 4 multiple-choice options and clearly mark the correct answer."
    elif qtype == "3":
        return "Answer strictly as True or False and give one-line justification."
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
