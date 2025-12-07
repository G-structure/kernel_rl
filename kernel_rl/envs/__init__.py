"""KernelBench RL environments."""

from kernel_rl.envs.kernelbench_client import (
    KernelBenchProblem,
    KernelEvalResult,
    evaluate_kernel,
    get_problem_ids,
    get_prompt_for_problem,
    get_reference_code,
)
from kernel_rl.envs.kernelbench_env import (
    KernelBenchDatasetBuilder,
    KernelBenchEnv,
    KernelBenchEnvGroupBuilder,
    KernelBenchRLDataset,
)
from kernel_rl.envs.multiturn_kernelbench_env import (
    MultiTurnKernelBenchDatasetBuilder,
    MultiTurnKernelBenchEnv,
    MultiTurnKernelBenchEnvGroupBuilder,
    MultiTurnKernelBenchRLDataset,
    MultiTurnState,
)

__all__ = [
    # Single-turn
    "KernelBenchProblem",
    "KernelEvalResult",
    "evaluate_kernel",
    "get_problem_ids",
    "get_prompt_for_problem",
    "get_reference_code",
    "KernelBenchDatasetBuilder",
    "KernelBenchEnv",
    "KernelBenchEnvGroupBuilder",
    "KernelBenchRLDataset",
    # Multi-turn (Kevin mode)
    "MultiTurnKernelBenchDatasetBuilder",
    "MultiTurnKernelBenchEnv",
    "MultiTurnKernelBenchEnvGroupBuilder",
    "MultiTurnKernelBenchRLDataset",
    "MultiTurnState",
]
