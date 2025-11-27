from loguru import logger
from openai import AzureOpenAI
import yaml
import os

def load_gpt(model_name):
    """Load an AzureOpenAI client.

    Priority: environment variables (OPENAI_API_KEY, OPENAI_BASE_URL[, OPENAI_API_VERSION])
    Fallback: `credentials.yml` with per-model entries.
    """
    # Environment variable override (recommended for CI / local runs)
    api_key = os.environ.get("OPENAI_API_KEY")
    api_version = os.environ.get("OPENAI_API_VERSION")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if api_key and base_url:
        logger.debug("Using OpenAI credentials from environment variables")
        model = AzureOpenAI(api_key=api_key, api_version=api_version, base_url=base_url)
        return model

    # Fallback to credentials file (legacy behavior)
    try:
        credentials = yaml.safe_load(open("credentials.yml"))
    except FileNotFoundError:
        raise FileNotFoundError(
            "No OPENAI_API_KEY/OPENAI_BASE_URL env vars found and credentials.yml missing. "
            "Provide credentials via env vars or create src/credentials.yml from src/credentials.example.yml."
        )

    assert model_name in credentials, f"Model {model_name} not found in credentials"

    credential = credentials[model_name]
    api_key = credential["api_key"]
    api_version = credential.get("api_version")
    base_url = credential["base_url"]

    model = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=base_url
    )

    return model


def call_gpt(model_name, model, openai_request):
    logger.debug(f"{model_name} is running")

    response = model.chat.completions.create(
        model=model_name, **openai_request
    )
    return response.choices[0].message.content

