"""
Reward shaping for KernelBench RL training.

This module implements reward functions that combine:
- Format correctness (valid code block extraction)
- Compilation success
- Correctness (passing all tests)
- Speed (optional, for later stages of training)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernel_rl.envs.kernelbench_client import KernelEvalResult


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Weights for different reward components
    format_weight: float = 0.1  # Reward for valid format
    compile_weight: float = 0.2  # Reward for successful compilation
    correctness_weight: float = 1.0  # Reward for passing tests
    speed_weight: float = 0.0  # Reward for speedup (disabled by default)

    # Penalties
    cheating_penalty: float = -1.0  # Penalty for just wrapping PyTorch
    format_penalty: float = -0.1  # Penalty for invalid format

    # Speed reward configuration
    speed_baseline: float = 1.0  # Speedup threshold for positive reward
    speed_scale: float = 0.5  # Scale factor for log speedup
    speed_max_reward: float = 1.0  # Maximum speed reward

    # Whether to use sparse rewards (only reward fully correct solutions)
    sparse_rewards: bool = False

    # Whether to use partial correctness rewards
    partial_correctness: bool = True


def format_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """
    Compute reward for format correctness.

    Returns:
        1.0 if format is valid (code block with correct language tag)
        config.format_penalty if format is invalid
    """
    if eval_result["format_ok"]:
        return 1.0
    return config.format_penalty


def compile_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """
    Compute reward for successful compilation.

    Returns:
        1.0 if compiled successfully and not cheating
        0.0 if compilation failed
        config.cheating_penalty if cheating detected
    """
    if eval_result["cheated"]:
        return config.cheating_penalty

    if eval_result["compiled"]:
        return 1.0

    return 0.0


def correctness_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """
    Compute reward for correctness.

    With partial_correctness=True:
        Returns tests_passed / tests_total (0.0 to 1.0)

    With partial_correctness=False:
        Returns 1.0 if all tests pass, 0.0 otherwise
    """
    if not eval_result["compiled"]:
        return 0.0

    tests_passed = eval_result["tests_passed"]
    tests_total = eval_result["tests_total"]

    if tests_total == 0:
        return 0.0

    if config.partial_correctness:
        return tests_passed / tests_total
    else:
        return 1.0 if tests_passed == tests_total else 0.0


def speed_reward(
    eval_result: "KernelEvalResult",
    config: RewardConfig,
    use_speed: bool = True
) -> float:
    """
    Compute reward for speedup over baseline.

    Only gives reward if:
    - use_speed is True
    - Kernel is fully correct
    - Speedup data is available

    The reward is based on log speedup:
        reward = scale * log2(speedup / baseline)

    Clamped to [0, max_reward].

    Args:
        eval_result: Evaluation result
        config: Reward configuration
        use_speed: Whether to use speed rewards

    Returns:
        Speed reward (0.0 if not applicable)
    """
    if not use_speed:
        return 0.0

    # Only reward speed for fully correct kernels
    if not eval_result["correctness"]:
        return 0.0

    speedup = eval_result.get("speedup")
    if speedup is None or speedup <= 0:
        return 0.0

    # Log-scaled reward
    if speedup <= config.speed_baseline:
        return 0.0

    log_speedup = math.log2(speedup / config.speed_baseline)
    reward = config.speed_scale * log_speedup

    # Clamp to max
    return min(reward, config.speed_max_reward)


def compute_reward(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
) -> float:
    """
    Compute the total reward for a kernel evaluation.

    The reward is a weighted combination of:
    - Format reward
    - Compile reward
    - Correctness reward
    - Speed reward (optional)

    With sparse_rewards=True, only fully correct solutions get reward.

    Args:
        eval_result: Result from kernel evaluation
        config: Reward configuration (uses defaults if None)

    Returns:
        Total reward (scalar)
    """
    if config is None:
        config = RewardConfig()

    # Sparse reward mode: only reward fully correct solutions
    if config.sparse_rewards:
        if eval_result["correctness"] and not eval_result["cheated"]:
            base_reward = 1.0
            # Add speed bonus if enabled
            if config.speed_weight > 0:
                s_reward = speed_reward(eval_result, config, use_speed=True)
                base_reward += config.speed_weight * s_reward
            return base_reward
        return 0.0

    # Dense reward mode: combine weighted components
    f_reward = format_reward(eval_result, config)
    c_reward = compile_reward(eval_result, config)
    corr_reward = correctness_reward(eval_result, config)
    s_reward = speed_reward(eval_result, config, use_speed=config.speed_weight > 0)

    total = (
        config.format_weight * f_reward
        + config.compile_weight * c_reward
        + config.correctness_weight * corr_reward
        + config.speed_weight * s_reward
    )

    return total


def compute_reward_breakdown(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
) -> dict[str, float]:
    """
    Compute individual reward components for logging.

    Returns:
        Dictionary with individual reward values
    """
    if config is None:
        config = RewardConfig()

    return {
        "reward_format": format_reward(eval_result, config),
        "reward_compile": compile_reward(eval_result, config),
        "reward_correctness": correctness_reward(eval_result, config),
        "reward_speed": speed_reward(eval_result, config, use_speed=True),
        "reward_total": compute_reward(eval_result, config),
    }


# Preset reward configurations for different training stages


def get_stage1_config() -> RewardConfig:
    """
    Stage 1: Focus on format and compilation.

    Good for early training when the model is learning
    to generate valid code structure.
    """
    return RewardConfig(
        format_weight=0.3,
        compile_weight=0.5,
        correctness_weight=0.2,
        speed_weight=0.0,
        partial_correctness=True,
        sparse_rewards=False,
    )


def get_stage2_config() -> RewardConfig:
    """
    Stage 2: Focus on correctness.

    For mid-training when the model can compile code
    but needs to learn correctness.
    """
    return RewardConfig(
        format_weight=0.1,
        compile_weight=0.2,
        correctness_weight=0.7,
        speed_weight=0.0,
        partial_correctness=True,
        sparse_rewards=False,
    )


def get_stage3_config() -> RewardConfig:
    """
    Stage 3: Correctness + Speed.

    For late training when the model can produce correct
    kernels and should optimize for speed.
    """
    return RewardConfig(
        format_weight=0.05,
        compile_weight=0.1,
        correctness_weight=0.5,
        speed_weight=0.35,
        partial_correctness=False,  # Only reward fully correct
        sparse_rewards=False,
    )


def get_sparse_config() -> RewardConfig:
    """
    Sparse rewards: Only reward fully correct solutions.

    May be useful for fine-tuning or when combined with
    curriculum learning.
    """
    return RewardConfig(
        sparse_rewards=True,
        speed_weight=0.5,
    )


REWARD_PRESETS = {
    "stage1": get_stage1_config,
    "stage2": get_stage2_config,
    "stage3": get_stage3_config,
    "sparse": get_sparse_config,
    "default": RewardConfig,
}


def get_reward_config(name: str) -> RewardConfig:
    """Get a reward configuration by name."""
    if name not in REWARD_PRESETS:
        raise ValueError(
            f"Unknown reward preset: {name}. Available: {list(REWARD_PRESETS.keys())}"
        )
    return REWARD_PRESETS[name]()
