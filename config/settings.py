# This prevents paths and model names from being scattered throughout the program.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIRECTORY = PROJECT_ROOT / "documents"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"
MANIFEST_PATH = PROJECT_ROOT / "data" / "document_manifest.json"
ENV_PATH = PROJECT_ROOT / ".env"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_COUNT = 4

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

WEB_SOURCES = [
    "https://verilator.org/guide/latest/",
    "https://verilator.org/guide/latest/exe_verilator.html",
    "https://verilator.org/guide/latest/verilating.html",
    "https://verilator.org/guide/latest/warnings.html",
    "https://yosyshq.readthedocs.io/projects/yosys/en/latest/",
    "https://chipverify.com/"
]