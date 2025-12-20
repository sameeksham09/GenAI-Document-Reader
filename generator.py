import ollama

def generate_answer(prompt, model="llama3.2:1b"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
