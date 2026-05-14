import os
import time
import httpx
import base64
import logging
from typing import List, Generator, Union
from google import genai
from openai import OpenAI
from google.genai.types import (GenerateContentConfig, HttpOptions, Tool, ToolCodeExecution)
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def _require(key: str) -> str:
    """Fetch a required env variable. Raises a clear error if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: '{key}'. Please set it in your .env file.")
    return value


# COIN access tokens typically expire in 60 min. Refresh the cached client
# well before that so a long-running RAG service never trips an auth error
# mid-request. Override at runtime with COIN_TOKEN_TTL_SECONDS.
COIN_TOKEN_TTL_SECONDS = int(os.getenv("COIN_TOKEN_TTL_SECONDS", str(14 * 60)))

# ── SSL / TLS ──────────────────────────────────────────────────────────────
SSL_CERT_FILE = _require("SSL_CERT_FILE")
os.environ["SSL_CERT_FILE"] = SSL_CERT_FILE
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_FILE

# ── COIN OAuth2 ────────────────────────────────────────────────────────────
COIN_URL = _require("COIN_URL")
COIN_CLIENT_ID = base64.b64decode(_require("COIN_CLIENT_ID").encode()).decode("utf-8")
COIN_CLIENT_SECRET = base64.b64decode(_require("COIN_CLIENT_SECRET").encode()).decode("utf-8")
COIN_SCOPE = base64.b64decode(_require("COIN_SCOPE").encode()).decode("utf-8")

# ── Vertex AI ──────────────────────────────────────────────────────────────
VERTEX_PROJECT = _require("VERTEX_PROJECT")
VERTEX_LOCATION = _require("VERTEX_LOCATION")
VERTEX_BASE_URL = _require("VERTEX_BASE_URL")
VERTEX_DEFAULT_MODEL = _require("VERTEX_DEFAULT_MODEL")
VERTEX_EMBEDDING_MODEL = _require("VERTEX_EMBEDDING_MODEL")

# ── Stellar ────────────────────────────────────────────────────────────────
STELLAR_BASE_URL = _require("STELLAR_BASE_URL")
STELLAR_CHAT_MODEL = _require("STELLAR_CHAT_MODEL")
STELLAR_MAX_TOKENS = int(_require("STELLAR_MAX_TOKENS"))


def get_coin_token() -> str:
    """
    Retrieves a COIN token by making a POST request to the COIN URL.

    Returns:
        str: The access token if the request is successful, otherwise None.
    """
    try:
        response = httpx.post(
            url=COIN_URL,
            verify=False,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={
                "grant_type": "client_credentials",
                "client_id": COIN_CLIENT_ID,
                "client_secret": COIN_CLIENT_SECRET,
                "scope": f"coinscope{COIN_SCOPE}",
            }
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


class VertexGenAI:
    """
    Vertex AI wrapper. Caches a `genai.Client` and rebuilds it every
    `ttl_seconds` so the underlying COIN token never expires from the
    caller's perspective. Pattern follows nlp2sql/agent/llm.py.
    """

    def __init__(self, ttl_seconds: int = COIN_TOKEN_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._client: "genai.Client | None" = None
        self._client_created: float = 0.0
        self.code_exec_tool = Tool(code_execution=ToolCodeExecution())
        self.default_model = VERTEX_DEFAULT_MODEL
        self.embedding_model = VERTEX_EMBEDDING_MODEL
        # Build eagerly so the first request isn't slower than the rest.
        self._build_client()

    def _build_client(self) -> None:
        token = get_coin_token()
        if not token:
            raise Exception("Failed to get COIN token for VertexGenAI.")
        credentials = Credentials(token)
        self._client = genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            http_options=HttpOptions(base_url=VERTEX_BASE_URL),
            location=VERTEX_LOCATION,
            credentials=credentials,
        )
        self._client_created = time.monotonic()
        logger.info("VertexGenAI client (re)built; TTL=%ds", self._ttl)

    @property
    def client(self):
        """Return the cached client, rebuilding it if the TTL has expired."""
        if self._client is None or (time.monotonic() - self._client_created) > self._ttl:
            self._build_client()
        return self._client

    def generate_content(self, query: str):
        response = self.client.models.generate_content(
            config=GenerateContentConfig(temperature=0),
            model=self.default_model,
            contents=query,
        )
        return response.text

    def create_embedding(self, query: Union[str, List[str]]):
        """
        Creates embeddings using the Vertex AI service. Handles single or batch inputs.
        """
        input_texts = [query] if isinstance(query, str) else query
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=input_texts,
        )
        embeddings = [e.values for e in response.embeddings]
        return embeddings[0] if isinstance(query, str) else embeddings


class StellarGenAI:
    """
    A client to interact with the OpenAI-compatible Stellar service.

    The underlying OpenAI client is rebuilt every `ttl_seconds` so a long-
    running service (e.g. the RAG backend) never runs into an expired COIN
    token mid-request. Existing code that touches `.client.foo.bar()` keeps
    working — `.client` is a property that does the refresh check.
    """

    def __init__(self, ttl_seconds: int = COIN_TOKEN_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._client: "OpenAI | None" = None
        self._client_created: float = 0.0
        self.chat_model = STELLAR_CHAT_MODEL
        self.max_tokens = STELLAR_MAX_TOKENS
        self.embedding_model_name = "gte-large-en-v1.5"
        # Build eagerly so first request isn't slower.
        self._build_client()
        print("StellarGenAI client initialized successfully.")

    def _build_client(self) -> None:
        token = get_coin_token()
        if not token:
            raise Exception("Failed to get COIN token for StellarGenAI.")
        self._client = OpenAI(
            api_key=token,
            base_url=STELLAR_BASE_URL,
            http_client=httpx.Client(verify=SSL_CERT_FILE),
        )
        self._client_created = time.monotonic()
        logger.info("StellarGenAI client (re)built; TTL=%ds", self._ttl)

    @property
    def client(self):
        """Return the cached client, rebuilding it if the TTL has expired."""
        if self._client is None or (time.monotonic() - self._client_created) > self._ttl:
            self._build_client()
        return self._client

    def create_embedding(self, query: Union[str, List[str]]):
        """
        Creates embeddings using the Stellar service. Handles single or batch inputs.
        """
        input_texts = [query] if isinstance(query, str) else query
        response = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=input_texts,
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if isinstance(query, str) else embeddings

    def generate_content_llama(self, query: str):
        """
        Generates content using the Stellar service with the configured chat model.
        """
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": query}],
            temperature=0,
        )
        return response.choices[0].message.content
