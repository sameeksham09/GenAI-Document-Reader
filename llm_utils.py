import os
import json
import requests

"""
LLM utility for the RAG system.

Supports three backends (set LLM_BACKEND):

1) ollama (default): local model via Ollama; no PyTorch in this process.
2) openai: hosted OpenAI API; needs OPENAI_API_KEY; may hit rate limits.
3) local_lora: your fine-tuned TinyLlama+LoRA from Training/lora-model.
   Uses PyTorch in this process; on some macOS setups this can crash
   (segfault). Use on Linux/Colab or after fixing your PyTorch install.

NEW IN THIS VERSION:
  generate_answer_stream() — yields tokens one by one for streaming UI.
  Only implemented for Ollama backend (most common local use case).
  OpenAI streaming can be added the same way using response.iter_lines().
"""


LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH  = os.path.join(os.path.dirname(__file__), "Training", "lora-model")

_local_lora_tokenizer = None
_local_lora_model     = None


# ── Non-streaming (used everywhere except main answer generation) ──────────────

def _generate_answer_ollama(prompt: str, max_new_tokens: int = 200) -> str:
    """Generate using a local Ollama model — returns full string."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": max_new_tokens,
                },
            },
            timeout=300,
        )
        if resp.status_code == 404:
            return (
                f"Ollama model not found: `{OLLAMA_MODEL}`.\n\n"
                f"Run:\n  ollama pull {OLLAMA_MODEL}\n"
            )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "Couldn't connect to Ollama.\n\n"
            "Install and start Ollama, then run:\n"
            f"  ollama pull {OLLAMA_MODEL}\n"
            "and retry.\n\n"
            f"(Expected Ollama at `{OLLAMA_HOST}`.)"
        )
    except Exception as e:
        return f"Error calling Ollama: {e}"


def _generate_answer_local_lora(prompt: str, max_new_tokens: int = 200) -> str:
    """Generate using your fine-tuned TinyLlama+LoRA."""
    global _local_lora_tokenizer, _local_lora_model

    if not os.path.isdir(LORA_PATH):
        return (
            f"Local LoRA path not found: {LORA_PATH}\n\n"
            "Train the adapter first."
        )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        return f"Local LoRA backend requires: pip install torch transformers peft\n\nImport error: {e}"

    device = torch.device("cpu")

    if _local_lora_model is None:
        try:
            _local_lora_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
            _local_lora_model = PeftModel.from_pretrained(base, LORA_PATH)
            _local_lora_model.to(device)
            _local_lora_model.eval()
        except Exception as e:
            return f"Error loading local LoRA model.\n\nDetails: {e}"

    try:
        inputs = _local_lora_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _local_lora_model.generate(**inputs, max_new_tokens=max_new_tokens)
        return _local_lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error running local LoRA model.\n\nDetails: {e}"


def _generate_answer_openai(prompt: str, max_new_tokens: int = 200) -> str:
    """Call OpenAI chat completion API."""
    if not OPENAI_API_KEY:
        return (
            "LLM backend is not configured.\n\n"
            "Please set the OPENAI_API_KEY environment variable.\n"
        )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":    OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.2,
            },
            timeout=60,
        )
        if resp.status_code == 429:
            return "OpenAI API rate limit hit. Please wait and retry."
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def generate_answer(prompt, max_new_tokens=200):
    """Public helper — returns full answer string. Used for rewrites, summaries, follow-ups."""
    if LLM_BACKEND == "openai":
        return _generate_answer_openai(prompt, max_new_tokens=max_new_tokens)
    if LLM_BACKEND == "local_lora":
        return _generate_answer_local_lora(prompt, max_new_tokens=max_new_tokens)
    return _generate_answer_ollama(prompt, max_new_tokens=max_new_tokens)


# ── Streaming ──────────────────────────────────────────────────────────────────

def _stream_ollama(prompt: str, max_new_tokens: int = 200):
    """
    Generator that yields tokens one by one from Ollama's streaming API.

    HOW OLLAMA STREAMING WORKS:
    ─────────────────────────────────────────────────────────────────────────
    When stream=True, Ollama does NOT wait to generate the full response.
    Instead it sends one JSON object per token over the HTTP connection:

        {"model":"llama3.2","response":"The","done":false}
        {"model":"llama3.2","response":" ACID","done":false}
        {"model":"llama3.2","response":" properties","done":false}
        ...
        {"model":"llama3.2","response":"","done":true}

    We use requests' stream=True mode which keeps the connection open and
    lets us iterate line by line with resp.iter_lines().

    For each line:
      - Parse the JSON
      - Extract the "response" field (one token)
      - yield it immediately to the caller

    When "done" is True, Ollama has finished generating — we stop.

    WHY yield INSTEAD OF return:
      yield makes this a Python generator. Instead of computing everything
      and returning a string, it returns one token at a time.
      The caller (st.write_stream) calls next() on this generator repeatedly,
      getting one token each time and displaying it immediately.
      This is what makes the text appear progressively in the UI.
    ─────────────────────────────────────────────────────────────────────────

    Args:
        prompt         : the full prompt string
        max_new_tokens : max tokens to generate

    Yields:
        One token string at a time
    """
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,              # ← KEY: enables token-by-token streaming
                "options": {
                    "temperature": 0.2,
                    "num_predict": max_new_tokens,
                },
            },
            stream=True,                     # ← tells requests: don't buffer the response
            timeout=300,
        )

        if resp.status_code == 404:
            yield f"Ollama model not found: `{OLLAMA_MODEL}`. Run: ollama pull {OLLAMA_MODEL}"
            return

        resp.raise_for_status()

        # iter_lines() gives us one line at a time as Ollama sends them
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data  = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token              # ← send one token to Streamlit immediately
                if data.get("done", False):
                    break                    # ← Ollama says it's finished
            except json.JSONDecodeError:
                continue

    except requests.exceptions.ConnectionError:
        yield (
            "Couldn't connect to Ollama. "
            f"Make sure Ollama is running and {OLLAMA_MODEL} is pulled."
        )
    except Exception as e:
        yield f"Error streaming from Ollama: {e}"


def _stream_openai(prompt: str, max_new_tokens: int = 200):
    """
    Generator that yields tokens from OpenAI's streaming API.

    OpenAI streaming uses Server-Sent Events (SSE) format:
        data: {"choices":[{"delta":{"content":"The"}}]}
        data: {"choices":[{"delta":{"content":" ACID"}}]}
        data: [DONE]

    Same pattern as Ollama — iterate lines, parse JSON, yield content.
    """
    if not OPENAI_API_KEY:
        yield "OpenAI API key not set. Set OPENAI_API_KEY environment variable."
        return

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":    OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.2,
                "stream":   True,            # ← OpenAI streaming flag
            },
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                data    = json.loads(line)
                content = data["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError):
                continue

    except Exception as e:
        yield f"Error streaming from OpenAI: {e}"


def generate_answer_stream(prompt: str, max_new_tokens: int = 200):
    """
    Public streaming interface — yields tokens one by one.

    Used in app_ui.py with st.write_stream():
        st.write_stream(generate_answer_stream(prompt))

    WHY SEPARATE FROM generate_answer():
        generate_answer() returns a string — used for:
          - Query rewriting (needs the full string to parse first line)
          - Document summarization
          - Smart follow-up buttons

        generate_answer_stream() yields tokens — used for:
          - Main answer display only (the part the user watches)

        Streaming doesn't work for rewriting because we need to inspect
        the full output (strip labels, take first line) before using it.
        You can't do that with a stream — you need the complete string first.

    Falls back to non-streaming for local_lora (PyTorch doesn't support
    easy token streaming without custom hooks).
    """
    if LLM_BACKEND == "openai":
        yield from _stream_openai(prompt, max_new_tokens=max_new_tokens)
    elif LLM_BACKEND == "local_lora":
        # LoRA doesn't support easy streaming — fall back to full generation
        # and yield it as a single chunk so the interface still works
        yield _generate_answer_local_lora(prompt, max_new_tokens=max_new_tokens)
    else:
        yield from _stream_ollama(prompt, max_new_tokens=max_new_tokens)


# ── Faithfulness Scoring ───────────────────────────────────────────────────────

def check_faithfulness(context, answer):
    """
    Check whether the generated answer is faithful to the retrieved context.

    WHAT IS FAITHFULNESS:
    ─────────────────────────────────────────────────────────────────────────
    Faithfulness measures whether every claim in the answer can be traced
    back to the retrieved context. An answer is unfaithful if it:
      - Introduces facts not present in the context (hallucination)
      - Contradicts what the context says
      - Makes up specific numbers, names, or definitions

    This is one of the four core RAGAs metrics:
      1. Faithfulness        ← what we're implementing
      2. Answer Relevance    (does the answer address the question?)
      3. Context Precision   (is the retrieved context relevant?)
      4. Context Recall      (does context cover the answer?)

    WHY A SECOND LLM CALL:
    ─────────────────────────────────────────────────────────────────────────
    We use the LLM as a judge — an "LLM-as-evaluator" pattern.
    The same model that generated the answer now reads both the context
    and the answer together and scores the overlap.

    This works because:
    - The judge sees BOTH at once — it can spot mismatches
    - It's language-native — understands paraphrasing vs hallucination
    - It's fast — a short structured prompt, max 80 tokens output

    Alternative: embedding similarity between answer and context.
    Problem: embeddings can't catch factual contradictions well.
    LLM judge is more accurate for faithfulness specifically.

    PROMPT DESIGN — WHY SO STRUCTURED:
    ─────────────────────────────────────────────────────────────────────────
    We ask for SCORE, VERDICT, REASON on separate lines.
    Small LLMs need very explicit format instructions.
    We parse the output by looking for these keywords line by line.
    If parsing fails, we return a safe "uncertain" default — never crash.

    Args:
        context : the retrieved chunks joined as a string (what was given to LLM)
        answer  : the generated answer string (what the LLM produced)

    Returns:
        dict with keys:
            score   : int 0-10 (10 = perfectly faithful)
            verdict : str "FAITHFUL" | "PARTIALLY FAITHFUL" | "NOT FAITHFUL"
            reason  : str one-line explanation
            emoji   : str ✅ | ⚠️ | ❌
    ─────────────────────────────────────────────────────────────────────────
    """

    # ── Build the judge prompt ────────────────────────────────────────────────
    # We cap context at 1500 chars to keep the prompt short.
    # The full context can be very long — we only need enough for the judge
    # to verify the answer claims. First 1500 chars covers the key content.
    context_preview = context[:1500]

    prompt = f"""You are evaluating whether an AI answer is faithful to the provided context.

Context (retrieved from document):
{context_preview}

Generated Answer:
{answer}

Score how faithful the answer is to the context above.
A faithful answer only contains information that appears in the context.
An unfaithful answer introduces facts, numbers, or claims not in the context.

Reply in EXACTLY this format (3 lines only, nothing else):
SCORE: [number from 0 to 10]
VERDICT: [FAITHFUL or PARTIALLY FAITHFUL or NOT FAITHFUL]
REASON: [one sentence explanation]"""

    # ── Call LLM ──────────────────────────────────────────────────────────────
    # We use generate_answer() (non-streaming) because we need the full
    # structured output to parse. Streaming would give us tokens one by one
    # which we can't parse until complete anyway.
    raw = generate_answer(prompt, max_new_tokens=80)

    # ── Parse the response ────────────────────────────────────────────────────
    # We parse line by line looking for SCORE:, VERDICT:, REASON: prefixes.
    # This is robust to extra whitespace or slight formatting variations.
    #
    # Default values if parsing fails — never crash the UI.
    score   = 5
    verdict = "UNCERTAIN"
    reason  = "Could not evaluate faithfulness."

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                # Extract the number after "SCORE:"
                # Handles "SCORE: 8", "SCORE:8", "SCORE: 8/10"
                num_str = line.split(":", 1)[1].strip().split("/")[0].strip()
                score   = max(0, min(10, int(float(num_str))))
            except:
                pass

        elif line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if "NOT" in v:
                verdict = "NOT FAITHFUL"
            elif "PARTIAL" in v:
                verdict = "PARTIALLY FAITHFUL"
            elif "FAITHFUL" in v:
                verdict = "FAITHFUL"

        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    # ── Map to emoji ──────────────────────────────────────────────────────────
    # Score thresholds:
    #   7-10 → FAITHFUL    ✅ (high confidence the answer came from context)
    #   4-6  → UNCERTAIN   ⚠️  (mixed — some claims may be unsupported)
    #   0-3  → NOT FAITHFUL ❌ (answer likely hallucinated)
    if score >= 7:
        emoji = "✅"
    elif score >= 4:
        emoji = "⚠️"
    else:
        emoji = "❌"

    return {
        "score":   score,
        "verdict": verdict,
        "reason":  reason,
        "emoji":   emoji,
    }