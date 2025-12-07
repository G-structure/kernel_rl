"""
Retrieval-Augmented In-Context Learning (RA-ICL) for kernel generation.

Uses KernelBook and AI-CUDA-Engineer-Archive as retrieval corpora.
"""

from kernel_rl.rag.corpus import KernelCorpus, KernelExample
from kernel_rl.rag.retriever import KernelRetriever
from kernel_rl.rag.prompt_builder import RAICLPromptBuilder

__all__ = [
    "KernelCorpus",
    "KernelExample",
    "KernelRetriever",
    "RAICLPromptBuilder",
]
