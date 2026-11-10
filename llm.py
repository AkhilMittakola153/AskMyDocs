import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "openrouter/free").strip()
# Optional extra models to try, comma-separated, e.g. FALLBACK_MODELS=google/gemma-3-27b-it:free
EXTRA = [m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()]
ROUTER = "openrouter/free"   # picks a random available free model on every call
ROUTER_TRIES = 3             # so retrying it usually lands on a different model

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY) if API_KEY else None


class LLMError(Exception):
    """Readable error to show in the app instead of a raw traceback."""


def _check_config():
    if not API_KEY:
        raise LLMError("OPENROUTER_API_KEY is missing. Add it to .env (local) or Secrets (Streamlit Cloud).")
    if "=" in MODEL or " " in MODEL:
        raise LLMError(f"MODEL looks wrong: '{MODEL}'. It should be like  MODEL=openrouter/free")


def _attempts():
    """Order of models to try: chosen model, extra fallbacks, then the free router several times."""
    order = list(dict.fromkeys([MODEL] + EXTRA))
    order += [ROUTER] * (ROUTER_TRIES - order.count(ROUTER))
    return order


def chat(system, user, max_tokens=3000):
    """Returns (answer_text, model_actually_used). Raises LLMError with a clear message."""
    _check_config()
    last_error = "unknown error"
    for model in _attempts():
        for attempt in range(2):  # one retry on rate limit
            try:
                r = client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=0,
                    # System text goes inside the user message: some free models reject the system role.
                    messages=[{"role": "user", "content": f"{system}\n\n---\nQuestion: {user}"}],
                    # Reasoning ("thinking") models spend output tokens on hidden reasoning first.
                    # Keep that short so tokens are left for the actual answer, and don't return it.
                    extra_body={"reasoning": {"effort": "low", "exclude": True}},
                )
                choices = getattr(r, "choices", None)
                used = getattr(r, "model", model)
                if not choices:
                    last_error = f"{used} returned no choices"
                    break
                text = (choices[0].message.content or "").strip()
                if text:
                    return text, used
                reason = choices[0].finish_reason
                last_error = (f"{used} used all {max_tokens} tokens on reasoning (finish_reason=length)"
                              if reason == "length" else f"{used} returned an empty reply ({reason})")
                break
            except APIStatusError as e:
                last_error = f"{model}: HTTP {e.status_code} - {e.message}"
                if e.status_code == 401:
                    raise LLMError("Invalid API key (401). Create a new key at openrouter.ai/keys.")
                if e.status_code == 429 and attempt == 0:
                    time.sleep(3)
                    continue
                break  # 400/404/402/5xx -> try the next model
            except APIConnectionError:
                raise LLMError("Can't reach openrouter.ai. Check your internet connection.")
    raise LLMError(f"All models failed. Last error: {last_error}")