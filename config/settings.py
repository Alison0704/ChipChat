from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIRECTORY = PROJECT_ROOT / "documents"
WEB_URLS_PATH = DOCUMENTS_DIRECTORY / "web_urls.txt"

PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"

CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"
MANIFEST_PATH = PROJECT_ROOT / "data" / "document_manifest.json"

ENV_PATH = PROJECT_ROOT / ".env"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_COUNT = 10

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

WEB_REQUEST_TIMEOUT = 20

WEB_USER_AGENT = (
    "ChipChat/1.0 "
    "(https://github.com/Alison0704/ChipChat)"
)