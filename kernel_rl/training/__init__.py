"""KernelBench RL training utilities."""

from kernel_rl.training.reward import (
    RewardConfig,
    compute_reward,
    compute_reward_breakdown,
    get_reward_config,
    REWARD_PRESETS,
    # Multi-turn (Kevin mode)
    MultiTurnRewardConfig,
    compute_discounted_returns,
    compute_trajectory_returns,
    compute_multiturn_rewards,
)
from kernel_rl.training.models import (
    ModelConfig,
    RECOMMENDED_MODELS,
    create_service_client,
    create_training_client,
    get_renderer_name_for_model,
    get_tokenizer_for_model,
)

# Note: loop and tensorboard_logger are imported lazily to avoid circular imports
# with kernel_rl.envs. Import directly from the modules:
#   from kernel_rl.training.loop import TrainingConfig, run_training_loop
#   from kernel_rl.training.tensorboard_logger import TensorBoardLogger, ...


def __getattr__(name: str):
    """Lazy import for modules that cause circular imports."""
    if name in ("TrainingConfig", "run_training_loop"):
        from kernel_rl.training.loop import TrainingConfig, run_training_loop
        if name == "TrainingConfig":
            return TrainingConfig
        return run_training_loop
    elif name in ("TensorBoardConfig", "TensorBoardLogger", "create_tensorboard_logger"):
        from kernel_rl.training.tensorboard_logger import (
            TensorBoardConfig,
            TensorBoardLogger,
            create_tensorboard_logger,
        )
        if name == "TensorBoardConfig":
            return TensorBoardConfig
        elif name == "TensorBoardLogger":
            return TensorBoardLogger
        return create_tensorboard_logger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Single-turn rewards
    "RewardConfig",
    "compute_reward",
    "compute_reward_breakdown",
    "get_reward_config",
    "REWARD_PRESETS",
    # Multi-turn rewards (Kevin mode)
    "MultiTurnRewardConfig",
    "compute_discounted_returns",
    "compute_trajectory_returns",
    "compute_multiturn_rewards",
    # Models
    "ModelConfig",
    "RECOMMENDED_MODELS",
    "create_service_client",
    "create_training_client",
    "get_renderer_name_for_model",
    "get_tokenizer_for_model",
    # Training loop
    "TrainingConfig",
    "run_training_loop",
    # TensorBoard
    "TensorBoardConfig",
    "TensorBoardLogger",
    "create_tensorboard_logger",
]
