from google import genai
from google.genai import types

from app.config import Settings
from app.core.errors import ConfigurationError


def create_genai_client(settings: Settings) -> genai.Client:
    api_key = settings.gemini_api_key_value
    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY is not configured")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1beta",
            timeout=int(settings.gemini_timeout_seconds * 1000),
        ),
    )
