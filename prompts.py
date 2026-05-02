# prompts.py

def get_instruction(qtype, num_questions=1):
    if qtype == "1":
        return "Answer the question clearly."
    elif qtype == "2":
        return f"Create {num_questions} multiple-choice questions and clearly mark the correct answers."
    elif qtype == "3":
        return f"Create {num_questions} True/False questions and provide one-line justification."
    elif qtype == "4":
        return "Create a fill-in-the-blank question and provide the answer."
    else:
        return None


STYLE_GUARD = """
Do not explain concepts beyond the document.
Do not introduce new definitions.
"""


def build_prompt(context, instruction, question, chat_history=None):
    """
    Build the final LLM prompt.
    Injects last 3 turns of conversation so the LLM has memory context.
    """
    history_block = ""
    if chat_history:
        recent = chat_history[-3:]
        lines  = ["Conversation so far (for context only — do NOT repeat this):"]
        for i, turn in enumerate(recent, 1):
            lines.append(f"  Turn {i}:")
            lines.append(f"    User asked : {turn['question']}")
            lines.append(f"    You answered: {turn['answer']}")
        history_block = "\n".join(lines) + "\n"

    return f"""
Use ONLY the context below.
If the answer can be inferred from the context, answer clearly in your own words.
Do NOT introduce information not present in the context.

{history_block}
Context:
{context}

Instruction:
{instruction}

{STYLE_GUARD}

Question:
{question}
"""


def build_rewrite_prompt(question, chat_history):
    """
    Rewrite the user's follow-up question into a standalone search query
    by resolving pronouns/references using conversation history.

    WHY THE PROMPT IS STRUCTURED THIS WAY:
    ────────────────────────────────────────────────────────────────────────
    Small LLMs (TinyLlama, qwen2.5:3b) have two failure modes with rewriting:

    Failure 1 — They echo the full history back as a numbered list.
      Fix: We only show PREVIOUS USER QUESTIONS (not answers) in the history.
           Shorter context = less for the model to pattern-match and echo.

    Failure 2 — They answer the question instead of rewriting it.
      Fix: Explicit rules ("ONE LINE only. No explanation. No numbering.")
           + a concrete few-shot example showing exact input→output format.
           Few-shot examples are the most reliable format-control for small LLMs.
    ────────────────────────────────────────────────────────────────────────
    """
    if not chat_history:
        return (
            "Rewrite this as a clean search query. "
            "ONE LINE only. No explanation. No numbering.\n\n"
            f"Question: {question}\n"
            "Rewritten:"
        )

    recent      = chat_history[-3:]
    topic_lines = [f"  Turn {i}: {t['question']}" for i, t in enumerate(recent, 1)]
    history_str = "\n".join(topic_lines)

    return (
        f"Previous questions in this conversation:\n"
        f"{history_str}\n\n"
        f"New question: \"{question}\"\n\n"
        f"Rewrite the new question as ONE standalone search query by resolving "
        f"any pronouns or vague references using the previous questions.\n"
        f"Rules: ONE LINE only. No numbering. No explanation. No punctuation at start.\n\n"
        f"Example:\n"
        f"  Previous: 'What are the ACID properties?'\n"
        f"  New question: 'Explain the second one'\n"
        f"  Rewritten: What is the Consistency property in ACID?\n\n"
        f"Rewritten:"
    )