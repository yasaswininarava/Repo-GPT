"""Limits, model names, and environment settings.

Embeddings stay on this computer. Only the answer request is sent to Gemini.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the project .env file. Missing files are fine; the app then explains
# which values the user still needs to add locally.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / ".cache" / "indexes"

# Lightweight CPU embedding model. all-MiniLM-L6-v2 accepts 256 tokens.
# Chunks are kept inside that limit so the model cannot silently truncate them.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MAX_INPUT_TOKENS = 256

# Verified on the Gemini pricing page on 2026-09-29: standard text input and
# output for this model were listed as "Free of charge". Google also recommends
# it for new projects. Quotas and free-tier eligibility can change.
# https://ai.google.dev/gemini-api/docs/pricing
# https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Official model page: input token limit 1,048,576. The app uses a much smaller
# budget so one question does not consume a large share of a free-tier quota.
# https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite
GEMINI_INPUT_TOKEN_LIMIT = 1_048_576
APP_MAX_INPUT_TOKENS = 8_000

TOP_K = 5
OVERLAP_LINES = 2
CHUNK_SETTINGS_VERSION = "v1"

MAX_FILE_BYTES = 150_000
MAX_TOTAL_BYTES = 3_000_000
MAX_FILES = 120
MAX_QUESTION_CHARS = 2_000
MAX_HISTORY_MESSAGES = 6
EXCERPT_CHAR_BUDGET = 12_000

SKIP_REASON_LABELS = {
    "dependency_directory": "dependency directories",
    "binary_or_media": "binary, image, or media files",
    "generated_file": "generated or lock files",
    "secret_file": "possible secret files",
    "too_large": "files over the size limit",
    "undecodable": "files that are not UTF-8 text",
    "file_cap": "files beyond the file-count limit",
    "byte_cap": "files beyond the total-size limit",
    "symlink": "symbolic links",
    "empty": "empty files",
}


def gemini_api_key() -> str | None:
    value = os.getenv("GEMINI_API_KEY", "").strip()
    return value or None


def gemini_model() -> str:
    value = os.getenv("GEMINI_MODEL", "").strip()
    return value or DEFAULT_GEMINI_MODEL


def github_token() -> str | None:
    value = os.getenv("GITHUB_TOKEN", "").strip()
    return value or None
