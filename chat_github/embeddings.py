"""Local CPU embeddings. This module never calls an embedding API."""

from __future__ import annotations

import os

import numpy as np

# Windows often cannot create the symlinks Hugging Face prefers. The model
# still downloads and runs; this only hides that warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from chat_github.config import EMBEDDING_MAX_INPUT_TOKENS, EMBEDDING_MODEL_NAME
from chat_github.errors import AppError, NetworkError


class SentenceTransformerEmbedder:
    """Wrap a small pretrained model so the rest of the app can stay simple."""

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AppError(
                "The sentence-transformers package is not installed. "
                "Create the virtual environment and run pip install -r requirements.txt."
            ) from exc
        try:
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        except OSError as exc:
            raise NetworkError(
                "The local embedding model could not be downloaded. "
                "Check your internet connection and try Load repository again. "
                "Embeddings are not sent to a paid API."
            ) from exc
        actual_limit = int(self._model.max_seq_length)
        if actual_limit != EMBEDDING_MAX_INPUT_TOKENS:
            raise AppError(
                f"The embedding model allows {actual_limit} tokens, but this app "
                f"expected {EMBEDDING_MAX_INPUT_TOKENS}. Refusing to continue so text "
                "is not silently truncated."
            )
        self.name = EMBEDDING_MODEL_NAME
        self.max_input_tokens = actual_limit

    def count_tokens(self, text: str) -> int:
        # Special tokens count toward the model limit, so they are included here.
        token_ids = self._model.tokenizer.encode(text, add_special_tokens=True)
        return len(token_ids)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._guard(texts)
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
        )
        return np.ascontiguousarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _guard(self, texts: list[str]) -> None:
        for text in texts:
            if self.count_tokens(text) > self.max_input_tokens:
                raise AppError(
                    "A chunk is longer than the embedding model's input limit. "
                    "It was not embedded, because the model would silently drop the end."
                )
