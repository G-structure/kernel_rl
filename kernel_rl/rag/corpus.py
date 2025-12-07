"""
Kernel corpus loaders for KernelBook and AI-CUDA-Engineer-Archive datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Literal

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


@dataclass
class KernelExample:
    """A single kernel example from the corpus."""

    pytorch_code: str
    kernel_code: str
    backend: Literal["triton", "cuda"]
    source: Literal["kernelbook", "sakana"]

    # Optional metadata
    name: str | None = None
    speedup: float | None = None
    correct: bool | None = None

    def __hash__(self) -> int:
        return hash((self.pytorch_code[:100], self.kernel_code[:100]))


class KernelCorpus:
    """
    Unified corpus from KernelBook (Triton) and AI-CUDA-Engineer-Archive (CUDA).

    Usage:
        corpus = KernelCorpus()
        corpus.load()  # Loads both datasets

        for example in corpus:
            print(example.pytorch_code, example.backend)
    """

    KERNELBOOK_REPO = "GPUMODE/KernelBook"
    SAKANA_REPO = "SakanaAI/AI-CUDA-Engineer-Archive"

    def __init__(
        self,
        include_kernelbook: bool = True,
        include_sakana: bool = True,
        sakana_levels: list[int] | None = None,
        sakana_correct_only: bool = True,
    ):
        """
        Initialize corpus configuration.

        Args:
            include_kernelbook: Include KernelBook (Triton) examples
            include_sakana: Include AI-CUDA-Engineer-Archive (CUDA) examples
            sakana_levels: Which Sakana levels to include (1, 2, 3). None = all
            sakana_correct_only: Only include correct kernels from Sakana
        """
        self.include_kernelbook = include_kernelbook
        self.include_sakana = include_sakana
        self.sakana_levels = sakana_levels or [1, 2, 3]
        self.sakana_correct_only = sakana_correct_only

        self._examples: list[KernelExample] = []
        self._loaded = False

    def load(self, cache_dir: str | None = None) -> None:
        """Load datasets from HuggingFace."""
        if self._loaded:
            return

        self._examples = []

        if self.include_kernelbook:
            self._load_kernelbook(cache_dir)

        if self.include_sakana:
            self._load_sakana(cache_dir)

        self._loaded = True
        logger.info(f"Loaded {len(self._examples)} kernel examples total")

    def _load_kernelbook(self, cache_dir: str | None) -> None:
        """Load KernelBook dataset (PyTorch -> Triton)."""
        logger.info(f"Loading KernelBook from {self.KERNELBOOK_REPO}")

        try:
            ds = load_dataset(self.KERNELBOOK_REPO, split="train", cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load KernelBook: {e}")
            return

        count = 0
        for row in ds:
            pytorch_code = row.get("python_code", "")
            triton_code = row.get("triton_code", "")

            if not pytorch_code or not triton_code:
                continue

            # Skip very long examples (likely too noisy)
            if len(pytorch_code) > 50000 or len(triton_code) > 100000:
                continue

            example = KernelExample(
                pytorch_code=pytorch_code,
                kernel_code=triton_code,
                backend="triton",
                source="kernelbook",
                name=row.get("entry_point") or row.get("module_name"),
            )
            self._examples.append(example)
            count += 1

        logger.info(f"Loaded {count} examples from KernelBook")

    def _load_sakana(self, cache_dir: str | None) -> None:
        """Load AI-CUDA-Engineer-Archive dataset (PyTorch -> CUDA)."""
        logger.info(f"Loading Sakana AI-CUDA-Engineer from {self.SAKANA_REPO}")

        try:
            ds = load_dataset(self.SAKANA_REPO, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load Sakana dataset: {e}")
            return

        count = 0
        for level in self.sakana_levels:
            split_name = f"level_{level}"
            if split_name not in ds:
                logger.warning(f"Split {split_name} not found in Sakana dataset")
                continue

            split_ds = ds[split_name]

            for row in split_ds:
                # Filter to correct kernels only if requested
                if self.sakana_correct_only and not row.get("Correct", False):
                    continue

                pytorch_code = row.get("PyTorch_Code_Module", "")
                cuda_code = row.get("CUDA_Code", "")

                if not pytorch_code or not cuda_code:
                    continue

                # Skip very long examples
                if len(pytorch_code) > 50000 or len(cuda_code) > 100000:
                    continue

                example = KernelExample(
                    pytorch_code=pytorch_code,
                    kernel_code=cuda_code,
                    backend="cuda",
                    source="sakana",
                    name=row.get("Op_Name") or row.get("Kernel_Name"),
                    speedup=row.get("CUDA_Speedup_Native"),
                    correct=row.get("Correct"),
                )
                self._examples.append(example)
                count += 1

        logger.info(f"Loaded {count} examples from Sakana")

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[KernelExample]:
        return iter(self._examples)

    def __getitem__(self, idx: int) -> KernelExample:
        return self._examples[idx]

    def filter_by_backend(self, backend: Literal["triton", "cuda"]) -> list[KernelExample]:
        """Get examples for a specific backend."""
        return [ex for ex in self._examples if ex.backend == backend]

    def get_pytorch_codes(self) -> list[str]:
        """Get all PyTorch codes (for embedding)."""
        return [ex.pytorch_code for ex in self._examples]
