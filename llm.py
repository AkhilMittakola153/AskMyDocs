import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "openrouter/free").strip()
# If the chosen model fails (removed, rate-limited, busy), fall back to OpenRouter's free router,
# which picks any currently available free model.
FALLBACKS = ["openrouter/free"]

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY) if API_KEY else None


class LLMError(Exception):
    """Readable error to show in the app instead of a raw traceback."""


def _check_config():
    if not API_KEY:
        raise LLMError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
    if "=" in MODEL or " " in MODEL:
        raise LLMError(f"MODEL in .env looks wrong: '{MODEL}'. It should be like  MODEL=openrouter/free")


def chat(system, user, max_tokens=1000):
    """Returns (answer_text, model_actually_used). Raises LLMError with a clear message."""
    _check_config()
    last_error = "unknown error"
    for model in dict.fromkeys([MODEL] + FALLBACKS):  # de-duplicated, order kept
        for attempt in range(2):                       # one retry on rate limit
            try:
                r = client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=0,
                    # System text goes inside the user message: some free models reject the system role.
                    messages=[{"role": "user", "content": f"{system}\n\n---\nQuestion: {user}"}],
                )
                choices = getattr(r, "choices", None)
                text = (choices[0].message.content or "").strip() if choices else ""
                if text:
                    return text, getattr(r, "model", model)
                last_error = f"{model} returned an empty reply"
                break
            except APIStatusError as e:
                last_error = f"{model}: HTTP {e.status_code} - {e.message}"
                if e.status_code == 401:
                    raise LLMError("Invalid API key (401). Create a new key at openrouter.ai/keys.")
                if e.status_code == 429 and attempt == 0:
                    time.sleep(3)
                    continue
                break                                   # 400/404/402/5xx -> try the next model
            except APIConnectionError:
                raise LLMError("Can't reach openrouter.ai. Check your internet connection.")
    raise LLMError(f"All models failed. Last error: {last_error}")