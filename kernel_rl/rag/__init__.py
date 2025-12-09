"""
Retrieval-Augmented In-Context Learning (RA-ICL) for kernel generation.

Uses KernelBook and AI-CUDA-Engineer-Archive as retrieval corpora.

Supports multiple embedding backends:
- MLX (Apple Silicon) - fastest on Mac
- CUDA (NVIDIA GPUs) - fastest on Linux/Windows with GPU
- MPS (Apple Metal via PyTorch) - fallback for Mac
- CPU - universal fallback
"""

from kernel_rl.rag.corpus import KernelCorpus, KernelExample
from kernel_rl.rag.embeddings import (
    BaseEmbedder,
    EmbeddingBackend,
    MLXEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    detect_best_backend,
)
from kernel_rl.rag.retriever import KernelRetriever
from kernel_rl.rag.prompt_builder import RAICLPromptBuilder

__all__ = [
    "KernelCorpus",
    "KernelExample",
    "KernelRetriever",
    "RAICLPromptBuilder",
    # Embedding backends
    "BaseEmbedder",
    "EmbeddingBackend",
    "MLXEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
    "detect_best_backend",
]
