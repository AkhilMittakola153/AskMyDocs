import os

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
    """Chosen model, extra fallbacks, then the free router several times."""
    order = list(dict.fromkeys([MODEL] + EXTRA))
    order += [ROUTER] * (ROUTER_TRIES - order.count(ROUTER))
    return order


def stream_chat(prompt, info, max_tokens=3000, temperature=0.2):
    """Yield the reply text piece by piece. If a model fails before replying, the next one is tried.
    info["model"] is set to the model that actually answered. Raises LLMError with a clear message."""
    _check_config()
    last_error = "unknown error"
    for model in _attempts():
        got_text = False
        try:
            stream = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=temperature, stream=True,
                # instructions go in the user message: some free models reject the system role
                messages=[{"role": "user", "content": prompt}],
                # keep hidden "thinking" short so tokens are left for the answer
                extra_body={"reasoning": {"effort": "low", "exclude": True}},
            )
            for chunk in stream:
                if getattr(chunk, "model", None):
                    info["model"] = chunk.model
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    got_text = True
                    yield delta
            if got_text:
                return
            last_error = f"{model} returned an empty reply"
        except APIStatusError as e:
            if e.status_code == 401:
                raise LLMError("Invalid API key (401). Create a new key at openrouter.ai/keys.")
            last_error = f"{model}: HTTP {e.status_code}"
        except APIConnectionError:
            raise LLMError("Can't reach openrouter.ai. Check your internet connection.")
        except Exception as e:  # stream cut off midway
            if got_text:
                raise LLMError("The reply was interrupted. Please ask again.")
            last_error = f"{model}: {e}"
    raise LLMError(f"All free models are busy right now. Please try again in a minute. ({last_error})")
