"""Local CPU embeddings. This module never calls an embedding API.

The model is still sentence-transformers/all-MiniLM-L6-v2. It runs through
ONNX Runtime instead of PyTorch, because Windows Application Control can
block PyTorch's DLL files and then no repository can be indexed.
"""

from __future__ import annotations

import os

import numpy as np

# Windows often cannot create the symlinks Hugging Face prefers. The model
# still downloads and runs; this only hides that warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from chat_github.config import EMBEDDING_MAX_INPUT_TOKENS, EMBEDDING_MODEL_NAME
from chat_github.errors import AppError, NetworkError

# Same weights as all-MiniLM-L6-v2, exported to ONNX. The public model name
# stored on each index stays EMBEDDING_MODEL_NAME.
_ONNX_REPO = "Qdrant/all-MiniLM-L6-v2-onnx"


class SentenceTransformerEmbedder:
    """Wrap the local MiniLM model so the rest of the app can stay simple."""

    def __init__(self) -> None:
        session, tokenizer = _load_runtime()
        self._session = session
        self._tokenizer = tokenizer
        self._input_names = {item.name for item in session.get_inputs()}
        self.name = EMBEDDING_MODEL_NAME
        self.max_input_tokens = EMBEDDING_MAX_INPUT_TOKENS

    def count_tokens(self, text: str) -> int:
        # Special tokens count toward the model limit, so they are included here.
        return len(self._tokenizer.encode(text).ids)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._guard(texts)
        encodings = self._tokenizer.encode_batch(texts)
        input_ids, attention_mask, token_type_ids = _pad(encodings)
        feeds = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = token_type_ids
        hidden = self._session.run(None, feeds)[0]
        pooled = _pool(hidden, attention_mask)
        return _normalize(pooled)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _guard(self, texts: list[str]) -> None:
        for text in texts:
            if self.count_tokens(text) > self.max_input_tokens:
                raise AppError(
                    "A chunk is longer than the embedding model's input limit. "
                    "It was not embedded, because the model would silently drop the end."
                )


def _load_runtime():
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise AppError(
            "The local embedding packages are not installed. "
            "Create the virtual environment and run pip install -r requirements.txt."
        ) from exc

    try:
        onnx_path = hf_hub_download(_ONNX_REPO, "model.onnx")
        tokenizer_path = hf_hub_download(_ONNX_REPO, "tokenizer.json")
    except OSError as exc:
        raise _download_error(exc) from exc
    except Exception as exc:
        if type(exc).__name__ in {"HfHubHTTPError", "LocalEntryNotFoundError"}:
            raise _download_error(exc) from exc
        raise

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.no_truncation()
    tokenizer.no_padding()

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, os.cpu_count() or 1)
    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except OSError as exc:
        if _policy_blocked(exc):
            raise AppError(
                "Windows blocked the local embedding library. "
                "Load repository cannot continue until that library is allowed to run."
            ) from exc
        raise
    return session, tokenizer


def _download_error(exc: BaseException) -> NetworkError:
    return NetworkError(
        "The local embedding model could not be downloaded. "
        "Check your internet connection and try Load repository again. "
        "Embeddings are not sent to a paid API."
    )


def _policy_blocked(exc: BaseException) -> bool:
    return getattr(exc, "winerror", None) == 4551 or "Application Control policy" in str(exc)


def _pad(encodings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(len(encoding.ids) for encoding in encodings)
    rows = len(encodings)
    input_ids = np.zeros((rows, width), dtype=np.int64)
    attention_mask = np.zeros((rows, width), dtype=np.int64)
    token_type_ids = np.zeros((rows, width), dtype=np.int64)
    for index, encoding in enumerate(encodings):
        count = len(encoding.ids)
        input_ids[index, :count] = encoding.ids
        attention_mask[index, :count] = encoding.attention_mask
        if encoding.type_ids:
            token_type_ids[index, :count] = encoding.type_ids
    return input_ids, attention_mask, token_type_ids


def _pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token vectors, which is what all-MiniLM-L6-v2 expects."""
    array = np.asarray(hidden, dtype=np.float32)
    if array.ndim == 2:
        return array
    mask = attention_mask.astype(np.float32)[..., None]
    summed = (array * mask).sum(axis=1)
    counts = np.maximum(mask.sum(axis=1), 1e-9)
    return summed / counts


def _normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.ascontiguousarray(array / np.maximum(norms, 1e-12))
