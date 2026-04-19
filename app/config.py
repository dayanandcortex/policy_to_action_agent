import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv(r"C:\Users\dayan\policy_to_action_agent\.env")


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return _get_ollama_llm()

    if provider == "openrouter":
        return _get_openrouter_llm()

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ----------------------------
# Ollama
# ----------------------------
def _get_ollama_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        temperature=0,
    )


# ----------------------------
# OpenRouter
# ----------------------------
def _get_openrouter_llm():
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")

    class OpenRouterWrapper:
        def invoke(self, prompt: str):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            return type("Obj", (), {
                "content": response.choices[0].message.content
            })

    return OpenRouterWrapper()

def get_max_chars_per_chunk() -> int:
    return get_env_int("MAX_CHARS_PER_CHUNK", 5000)


def get_max_revisions() -> int:
    return get_env_int("MAX_REVISIONS", 2)