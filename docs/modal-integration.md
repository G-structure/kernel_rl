# Modal Labs Integration

This document explains how KernelBench RL uses [Modal Labs](https://modal.com) for isolated, parallel GPU kernel evaluation.

## Overview

Modal provides:
- **Process isolation**: Each kernel runs in its own container
- **Hard timeouts**: Bad kernels are killed reliably at the container level
- **Parallel execution**: Evaluate many kernels concurrently across multiple GPUs
- **Fresh GPU state**: No cross-contamination between kernel evaluations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Local Training Loop                          │
│  ┌─────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │   Tinker    │───▶│  Kernel Code    │───▶│    Modal       │  │
│  │   (LLM)     │    │   Generation    │    │   Evaluator    │  │
│  └─────────────┘    └─────────────────┘    └───────┬────────┘  │
└──────────────────────────────────────────────────────┼──────────┘
                                                       │
                        ┌──────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Modal Cloud Infrastructure                    │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │  Container 1 │  │  Container 2 │  │  Container N │   ...   │
│   │   + A100     │  │   + A100     │  │   + A100     │         │
│   │              │  │              │  │              │         │
│   │  Kernel #1   │  │  Kernel #2   │  │  Kernel #N   │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install Modal

```bash
uv add modal
```

### 2. Authenticate with Modal

```bash
modal token new
```

This will open a browser for authentication and save credentials to `~/.modal.toml`.

Alternatively, set environment variables:
```bash
export MODAL_TOKEN_ID="your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"
```

### 3. Deploy the Modal App

**This step is required before running training with Modal.**

```bash
modal deploy kernel_rl/modal/app.py
```

You should see output like:
```
✓ Created objects.
├── 🔨 Created mount /path/to/app.py
├── 🔨 Created mount PythonPackage:src
└── 🔨 Created function KernelEvaluator.*.
✓ App deployed! 🎉

View Deployment: https://modal.com/apps/your-org/main/deployed/kernel-rl-evaluator
```

### 4. Enable Modal in Config

In your training config (`kernel_rl/config/rl_kernelbench.yaml`):

```yaml
dataset_builder:
  use_modal: true
  modal_gpu_type: "A100"  # Options: A100, H100, L40S, T4, etc.
  modal_timeout: 180.0    # Timeout per kernel in seconds
```

## How It Works

### Deployed vs Ephemeral Mode

The evaluator uses **deployed mode** by default:

1. **Deployed Mode** (recommended):
   - Uses a pre-deployed Modal app via `modal.Cls.from_name()`
   - No cold boot overhead - functions are always ready
   - Supports unlimited concurrent evaluations
   - Requires running `modal deploy` first

2. **Ephemeral Mode** (legacy, not recommended):
   - Spins up Modal app on-demand with `app.run()`
   - Has cold boot overhead (~10-30 seconds first call)
   - Cannot handle concurrent evaluations properly

### Calling Deployed Functions

When you call `evaluator.evaluate.remote(...)`:

```python
# The evaluator is looked up by name from Modal's registry
evaluator_cls = modal.Cls.from_name("kernel-rl-evaluator", "KernelEvaluator")

# Each .remote() call gets its own isolated container + GPU
result = evaluator_cls().evaluate.remote(
    ref_code=ref_code,
    kernel_code=kernel_code,
    ...
)
```

### Batch Evaluation with starmap

For evaluating many kernels in parallel:

```python
# Prepare list of argument tuples
args = [(ref1, kernel1, ...), (ref2, kernel2, ...), ...]

# starmap dispatches all calls in parallel across Modal's infrastructure
results = list(evaluator.evaluate.starmap(args, return_exceptions=True))
```

Modal automatically:
- Schedules work across available GPUs
- Handles container lifecycle
- Returns results in order

## GPU Options

Configure GPU type in your training config:

| GPU Type | Modal Name | Architecture | Best For |
|----------|------------|--------------|----------|
| A100-40GB | `"A100"` | Ampere | Production training |
| A100-80GB | `"A100-80GB"` | Ampere | Large kernels |
| H100 | `"H100"` | Hopper | Latest features |
| L40S | `"L40S"` | Ada | Cost-effective |
| T4 | `"T4"` | Turing | Development/testing |

## Timeout Handling

Modal enforces hard timeouts at the container level:

```python
# In app.py
@app.cls(image=image)
class KernelEvaluator:
    @modal.method()
    def evaluate(self, ...):
        # This method has a configurable timeout
        ...

# Timeout is set via .with_options()
evaluator_cls = KernelEvaluator.with_options(
    gpu=gpu_type,
    timeout=timeout_seconds
)
```

If a kernel hangs or runs too long:
1. Modal terminates the container after `timeout` seconds
2. The call raises a timeout exception
3. Other concurrent evaluations are unaffected

## Monitoring

### View Active Containers

```bash
modal container list
```

### View Logs

```bash
modal app logs kernel-rl-evaluator
```

### Check Deployment Status

```bash
modal app list
```

Or visit the Modal dashboard: https://modal.com/apps

## Troubleshooting

### "App is already running" Error

This happens when using ephemeral mode (`app.run()`) with concurrent calls.

**Solution**: Deploy the app first:
```bash
modal deploy kernel_rl/modal/app.py
```

### "Modal app not found" Error

The app hasn't been deployed yet.

**Solution**: Run:
```bash
modal deploy kernel_rl/modal/app.py
```

### Authentication Errors

Token is missing or invalid.

**Solution**:
```bash
# Re-authenticate
modal token new

# Or set environment variables
export MODAL_TOKEN_ID="..."
export MODAL_TOKEN_SECRET="..."
```

### Slow First Evaluation

Cold boot time for new containers.

**Solutions**:
- Use deployed mode (already configured)
- Modal warms containers after first use
- Consider using cheaper GPUs (T4) for development

### Kernel Compilation Errors on Modal

The Modal container may have different CUDA/compiler versions.

**Check**:
- Modal image uses CUDA 12.8
- Ensure your kernel code doesn't have hardcoded paths
- GPU architecture is passed correctly (Ampere, Hopper, etc.)

## Cost Optimization

1. **Use appropriate GPU types**: T4 for development, A100 for production
2. **Batch evaluations**: Use `starmap()` instead of many individual `.remote()` calls
3. **Set reasonable timeouts**: Don't wait 3 minutes for a kernel that's clearly hanging

## Files Reference

| File | Purpose |
|------|---------|
| `kernel_rl/modal/app.py` | Modal app definition, container image, GPU config |
| `kernel_rl/modal/evaluator.py` | High-level evaluator interface, batch processing |
| `kernel_rl/modal/__init__.py` | Public exports |

## Example Usage

### Basic Single Evaluation

```python
from kernel_rl.modal import ModalKernelEvaluator, ModalEvaluatorConfig

config = ModalEvaluatorConfig(
    gpu_type="A100",
    timeout=180,
)
evaluator = ModalKernelEvaluator(config)

result = await evaluator.evaluate_single(
    ref_code="def forward(x): return x * 2",
    kernel_code="...",
    backend="triton",
)
print(result)
# {'compiled': True, 'correctness': True, 'speedup': 1.5, ...}
```

### Batch Evaluation

```python
evaluations = [
    {"ref_code": ref1, "kernel_code": kernel1, "backend": "triton"},
    {"ref_code": ref2, "kernel_code": kernel2, "backend": "triton"},
    # ... more kernels
]

results = await evaluator.evaluate_batch(evaluations)
# Returns list of results in same order as input
```

### Integration with Training Loop

The training loop automatically uses Modal when `use_modal: true` is set:

```python
# In multiturn_kernelbench_env.py
if self.use_modal:
    from kernel_rl.modal.evaluator import get_modal_evaluator
    evaluator = get_modal_evaluator(config)
    result = await evaluator.evaluate_single(...)
```
