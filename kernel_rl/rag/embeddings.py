"""
Embedding backends for RAG retrieval.

Supports multiple backends:
- MLX (Apple Silicon) - fastest on Mac
- sentence-transformers with CUDA/MPS/CPU - cross-platform
- Auto-detection based on available hardware
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

EmbeddingBackend = Literal["mlx", "cuda", "mps", "cpu", "auto"]


def detect_best_backend() -> EmbeddingBackend:
    """Auto-detect the best available embedding backend."""
    # Check for explicit override
    env_device = os.environ.get("RAG_DEVICE", "").lower()
    if env_device in ("mlx", "cuda", "mps", "cpu"):
        return env_device

    # Try MLX first (best for Apple Silicon)
    try:
        import mlx.core as mx

        # Verify MLX is functional
        _ = mx.array([1.0])
        logger.info("Using MLX backend for embeddings")
        return "mlx"
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"MLX available but not functional: {e}")

    # Try CUDA
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Using CUDA backend for embeddings")
            return "cuda"
    except ImportError:
        pass

    # Try MPS (Apple Metal via PyTorch)
    try:
        import torch

        if torch.backends.mps.is_available():
            logger.info("Using MPS backend for embeddings")
            return "mps"
    except ImportError:
        pass

    # Fallback to CPU
    logger.info("Using CPU backend for embeddings")
    return "cpu"


class BaseEmbedder(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to normalized embeddings.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress

        Returns:
            Normalized embeddings as numpy array of shape (len(texts), dim)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding backend using sentence-transformers (CUDA/MPS/CPU)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-code-v1",
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._dim


class MLXEmbedder(BaseEmbedder):
    """
    Embedding backend using MLX for Apple Silicon.

    Uses mlx-embedding-models library for native MLX BERT embeddings.
    https://pypi.org/project/mlx-embedding-models/
    """

    # Default model from the mlx-embedding-models registry
    DEFAULT_MODEL = "bge-small"

    def __init__(
        self,
        model_name: str = "bge-small",
    ):
        # Load HF token from environment for private models
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except Exception as e:
                logger.debug(f"HF login failed: {e}")

        from mlx_embedding_models.embedding import EmbeddingModel

        self._model = EmbeddingModel.from_registry(model_name)
        # BGE-small has 384-dim embeddings
        self._dim = 384
        logger.info(f"Loaded MLX embedding model: {model_name}")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        # mlx-embedding-models handles batching internally
        embeddings = self._model.encode(texts)
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._dim


def create_embedder(
    model_name: str = "BAAI/bge-code-v1",
    backend: EmbeddingBackend = "auto",
) -> BaseEmbedder:
    """
    Create an embedder with the specified backend.

    Args:
        model_name: Model name for sentence-transformers (used as reference for MLX)
        backend: Backend to use ("mlx", "cuda", "mps", "cpu", or "auto")

    Returns:
        Configured embedder instance
    """
    if backend == "auto":
        backend = detect_best_backend()

    if backend == "mlx":
        try:
            return MLXEmbedder(model_name)
        except ImportError:
            logger.warning("MLX not installed, falling back to sentence-transformers")
            backend = "cpu"

    # Use sentence-transformers for cuda/mps/cpu
    device = None if backend == "cpu" else backend
    return SentenceTransformerEmbedder(model_name, device=device)
