import os
import requests

"""
LLM utility for the RAG system.

Supports three backends (set LLM_BACKEND):

1) ollama (default): local model via Ollama; no PyTorch in this process.
2) openai: hosted OpenAI API; needs OPENAI_API_KEY; may hit rate limits.
3) local_lora: your fine-tuned TinyLlama+LoRA from Training/lora-model.
   Uses PyTorch in this process; on some macOS setups this can crash
   (segfault). Use on Linux/Colab or after fixing your PyTorch install.
"""


LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Fine-tuned LoRA (only used when LLM_BACKEND=local_lora)
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = os.path.join(os.path.dirname(__file__), "Training", "lora-model")

# Cached model/tokenizer for local_lora (loaded once)
_local_lora_tokenizer = None
_local_lora_model = None


def _generate_answer_ollama(prompt: str, max_new_tokens: int = 200) -> str:
    """Generate using a local Ollama model."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": max_new_tokens,
                },
            },
            timeout=120,
        )
        if resp.status_code == 404:
            return (
                f"Ollama model not found: `{OLLAMA_MODEL}`.\n\n"
                f"Run:\n  ollama pull {OLLAMA_MODEL}\n"
            )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
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
    """Generate using your fine-tuned TinyLlama+LoRA from Training/lora-model."""
    global _local_lora_tokenizer, _local_lora_model

    if not os.path.isdir(LORA_PATH):
        return (
            f"Local LoRA path not found: {LORA_PATH}\n\n"
            "Train the adapter first: run create_training_dataset.py, "
            "convert_to_instruction.py, then Training/train_lora.py (e.g. on Colab), "
            "and place the output in Training/lora-model."
        )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        return (
            "Local LoRA backend requires: pip install torch transformers peft\n\n"
            f"Import error: {e}"
        )

    device = torch.device("cpu")

    if _local_lora_model is None:
        try:
            _local_lora_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
            )
            _local_lora_model = PeftModel.from_pretrained(base, LORA_PATH)
            _local_lora_model.to(device)
            _local_lora_model.eval()
        except Exception as e:
            return (
                "Error loading local LoRA model. On some Macs PyTorch can crash (segfault); "
                "try LLM_BACKEND=ollama or run the app on Linux/Colab.\n\n"
                f"Details: {e}"
            )

    try:
        inputs = _local_lora_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _local_lora_model.generate(**inputs, max_new_tokens=max_new_tokens)
        return _local_lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return (
            "Error running local LoRA model.\n\n"
            f"Details: {e}"
        )


def _generate_answer_openai(prompt: str, max_new_tokens: int = 200) -> str:
    """Call an OpenAI-compatible chat completion API for generation."""
    if not OPENAI_API_KEY:
        return (
            "LLM backend is not configured.\n\n"
            "Please set the OPENAI_API_KEY environment variable in your shell, e.g.:\n"
            "  export OPENAI_API_KEY='sk-...'\n\n"
            "Optionally set OPENAI_MODEL (default: gpt-4o-mini).\n"
        )

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0.2,
            },
            timeout=60,
        )

        # Friendly handling for common HTTP errors
        if resp.status_code == 429:
            return (
                "The OpenAI API is currently rate-limiting this key "
                "(HTTP 429: Too Many Requests).\n\n"
                "Please wait a bit and try again, or check your usage/limits "
                "on the OpenAI dashboard. If you're on a free/limited tier, "
                "you may need to reduce the number of requests or upgrade your plan."
            )

        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def generate_answer(prompt, max_new_tokens=200):
    """Public helper used by the rest of the app."""
    if LLM_BACKEND == "openai":
        return _generate_answer_openai(prompt, max_new_tokens=max_new_tokens)
    if LLM_BACKEND == "local_lora":
        return _generate_answer_local_lora(prompt, max_new_tokens=max_new_tokens)
    return _generate_answer_ollama(prompt, max_new_tokens=max_new_tokens)
