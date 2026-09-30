"""Stand-in embedder for local tests. It does not download a model."""

from __future__ import annotations

import numpy as np


class HashEmbedder:
    """Map a few known words onto separate dimensions so search is deterministic."""

    def __init__(self) -> None:
        self.name = "hash-test"
        self.max_input_tokens = 64

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(4, dtype=np.float32)
        lowered = text.lower()
        if "uniquealpha" in lowered:
            vector[0] = 1.0
        if "uniquebeta" in lowered:
            vector[1] = 1.0
        if not vector.any():
            vector[2] = 1.0
        vector /= np.linalg.norm(vector)
        return vector
