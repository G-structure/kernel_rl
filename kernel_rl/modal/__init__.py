"""
Modal Labs integration for isolated GPU kernel evaluation.

This module provides Modal-based kernel evaluation with:
- Hard timeout enforcement (kills bad kernels)
- Process isolation (each kernel in separate container)
- Parallel batch evaluation across multiple GPUs
"""

from kernel_rl.modal.evaluator import (
    ModalEvaluatorConfig,
    ModalKernelEvaluator,
    evaluate_kernel_batch,
    evaluate_kernel_single,
)

__all__ = [
    "ModalEvaluatorConfig",
    "ModalKernelEvaluator",
    "evaluate_kernel_batch",
    "evaluate_kernel_single",
]
