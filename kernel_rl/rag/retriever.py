"""
KISS retriever for kernel examples using FAISS and sentence-transformers.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


class KernelRetriever:
    """
    Simple retriever using sentence-transformers embeddings + FAISS index.

    Usage:
        # Build index (once)
        corpus = KernelCorpus()
        corpus.load()
        retriever = KernelRetriever()
        retriever.build_index(corpus)
        retriever.save("kernel_index")

        # Use index (at runtime)
        retriever = KernelRetriever.load("kernel_index")
        examples = retriever.retrieve(pytorch_code, k=3, backend="triton")
    """

    DEFAULT_MODEL = "microsoft/codebert-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ):
        """
        Initialize retriever.

        Args:
            model_name: Sentence-transformer model for embeddings
            device: Device for embedding model ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name
        self.device = device

        self._model = None
        self._index = None
        self._examples: list = []
        self._embeddings: np.ndarray | None = None

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )
        return embeddings

    def build_index(
        self,
        corpus,
        batch_size: int = 32,
    ) -> None:
        """
        Build FAISS index from corpus.

        Args:
            corpus: KernelCorpus instance (must be loaded)
            batch_size: Batch size for embedding
        """
        import faiss

        logger.info(f"Building index for {len(corpus)} examples")

        # Store examples
        self._examples = list(corpus)

        # Get PyTorch codes for embedding
        pytorch_codes = corpus.get_pytorch_codes()

        # Truncate very long codes for embedding
        max_len = 8000  # Most embedding models have token limits
        pytorch_codes = [code[:max_len] for code in pytorch_codes]

        # Embed
        logger.info("Embedding PyTorch codes...")
        self._embeddings = self._embed(pytorch_codes, batch_size=batch_size)

        # Build FAISS index (inner product = cosine since normalized)
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings.astype(np.float32))

        logger.info(f"Built index with {self._index.ntotal} vectors, dim={dim}")

    def retrieve(
        self,
        query_code: str,
        k: int = 3,
        backend: Literal["triton", "cuda"] | None = None,
    ) -> list[tuple[float, "KernelExample"]]:
        """
        Retrieve top-k similar kernel examples.

        Args:
            query_code: PyTorch code to find examples for
            k: Number of examples to retrieve
            backend: Filter to specific backend (None = any)

        Returns:
            List of (score, KernelExample) tuples, sorted by similarity
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        # Embed query
        query_code = query_code[:8000]  # Truncate
        query_emb = self._embed([query_code])[0:1].astype(np.float32)

        # Search (get more if filtering by backend)
        search_k = k * 3 if backend else k
        scores, indices = self._index.search(query_emb, min(search_k, len(self._examples)))

        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue

            example = self._examples[idx]

            # Filter by backend if specified
            if backend and example.backend != backend:
                continue

            results.append((float(score), example))

            if len(results) >= k:
                break

        return results

    def save(self, path: str | Path) -> None:
        """Save index and examples to disk."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(path / "index.faiss"))

        # Save examples (pickle for simplicity)
        with open(path / "examples.pkl", "wb") as f:
            pickle.dump(self._examples, f)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_examples": len(self._examples),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved index to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "KernelRetriever":
        """Load index and examples from disk."""
        import faiss

        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Create retriever
        retriever = cls(model_name=metadata["model_name"], device=device)

        # Load FAISS index
        retriever._index = faiss.read_index(str(path / "index.faiss"))

        # Load examples
        with open(path / "examples.pkl", "rb") as f:
            retriever._examples = pickle.load(f)

        logger.info(
            f"Loaded index with {retriever._index.ntotal} vectors, "
            f"{len(retriever._examples)} examples"
        )

        return retriever
