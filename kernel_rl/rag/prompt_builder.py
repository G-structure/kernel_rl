"""
RA-ICL prompt builder that integrates retrieved examples into prompts.
"""

from __future__ import annotations

import logging
from typing import Literal

from kernel_rl.rag.corpus import KernelExample
from kernel_rl.rag.retriever import KernelRetriever

logger = logging.getLogger(__name__)


RAICL_SYSTEM_PROMPT = """You are an expert GPU kernel developer. Your task is to optimize PyTorch operations by writing efficient custom GPU kernels.

You will be shown examples of similar PyTorch code and their optimized kernel implementations. Use these examples to guide your optimization approach, but adapt the techniques to the specific problem at hand.

When given a PyTorch model, you should:
1. Analyze the operations being performed
2. Study the provided examples for relevant optimization patterns
3. Write an optimized kernel implementation
4. Return your solution as a Python class named `ModelNew`

Your solution must:
- Be a drop-in replacement (same inputs/outputs as the original)
- Use custom {backend} kernels, not just PyTorch operations
- Be correct and produce the same results as the reference

You MUST respond in exactly this format:

<think>
1-5 short bullet points describing:
- What optimization strategy you will use (based on the examples)
- Key implementation details (tiling, memory layout, etc.)
- Any constraints or edge cases to handle

Keep this section under 150 tokens.
</think>

<KERNEL>
```python
# Your complete optimized implementation here
class ModelNew(nn.Module):
    ...
```
</KERNEL>
"""


EXAMPLE_TEMPLATE = """
### Example {idx}: {name}

**PyTorch Implementation:**
```python
{pytorch_code}
```

**Optimized {backend} Implementation:**
```python
{kernel_code}
```
"""


PROBLEM_TEMPLATE = """
## Your Task

Optimize the following PyTorch model using custom {backend} kernels:

```python
{ref_code}
```

Write your optimized implementation as a class named `ModelNew`.

Remember to use the required format: <think>...</think> followed by <KERNEL>...</KERNEL>.
"""


class RAICLPromptBuilder:
    """
    Builds prompts with retrieved in-context examples.

    Usage:
        retriever = KernelRetriever.load("kernel_index")
        builder = RAICLPromptBuilder(retriever)

        prompt = builder.build_prompt(
            ref_code=pytorch_code,
            backend="triton",
            k=3,
        )
    """

    def __init__(
        self,
        retriever: KernelRetriever,
        max_example_len: int = 10000,
    ):
        """
        Initialize prompt builder.

        Args:
            retriever: Loaded KernelRetriever instance
            max_example_len: Max chars per example (truncate long ones)
        """
        self.retriever = retriever
        self.max_example_len = max_example_len

    def build_system_prompt(self, backend: str) -> str:
        """Build the system prompt."""
        return RAICL_SYSTEM_PROMPT.format(backend=backend.upper())

    def build_prompt(
        self,
        ref_code: str,
        backend: Literal["triton", "cuda"],
        k: int = 3,
        include_system: bool = False,
    ) -> str:
        """
        Build a prompt with retrieved examples.

        Args:
            ref_code: Reference PyTorch code to optimize
            backend: Target backend ("triton" or "cuda")
            k: Number of examples to retrieve
            include_system: Whether to include system prompt

        Returns:
            Complete prompt string
        """
        # Retrieve similar examples
        results = self.retriever.retrieve(ref_code, k=k, backend=backend)

        # Build prompt parts
        parts = []

        # Optional system prompt
        if include_system:
            parts.append(self.build_system_prompt(backend))
            parts.append("\n---\n")

        # Add examples section
        if results:
            parts.append("## Similar Optimization Examples\n")
            parts.append("Study these examples of PyTorch to kernel transformations:\n")

            for idx, (score, example) in enumerate(results, 1):
                example_text = self._format_example(example, idx, backend)
                parts.append(example_text)

            parts.append("\n---\n")

        # Add the actual problem
        parts.append(PROBLEM_TEMPLATE.format(
            backend=backend.upper(),
            ref_code=ref_code,
        ))

        return "".join(parts)

    def _format_example(
        self,
        example: KernelExample,
        idx: int,
        backend: str,
    ) -> str:
        """Format a single example for the prompt."""
        # Truncate long code
        pytorch_code = example.pytorch_code
        kernel_code = example.kernel_code

        if len(pytorch_code) > self.max_example_len:
            pytorch_code = pytorch_code[: self.max_example_len] + "\n# ... (truncated)"

        if len(kernel_code) > self.max_example_len:
            kernel_code = kernel_code[: self.max_example_len] + "\n# ... (truncated)"

        name = example.name or f"{example.source}_{idx}"

        return EXAMPLE_TEMPLATE.format(
            idx=idx,
            name=name,
            pytorch_code=pytorch_code.strip(),
            kernel_code=kernel_code.strip(),
            backend=backend.upper(),
        )

    def get_messages(
        self,
        ref_code: str,
        backend: Literal["triton", "cuda"],
        k: int = 3,
    ) -> list[dict[str, str]]:
        """
        Get prompt as message list (for chat models).

        Returns:
            List of {"role": ..., "content": ...} dicts
        """
        return [
            {"role": "system", "content": self.build_system_prompt(backend)},
            {"role": "user", "content": self.build_prompt(ref_code, backend, k, include_system=False)},
        ]


def get_raicl_prompt(
    ref_code: str,
    backend: Literal["triton", "cuda"],
    retriever: KernelRetriever,
    k: int = 3,
) -> str:
    """
    Convenience function to get a RA-ICL prompt.

    Args:
        ref_code: Reference PyTorch code
        backend: Target backend
        retriever: Loaded retriever instance
        k: Number of examples

    Returns:
        Complete prompt string
    """
    builder = RAICLPromptBuilder(retriever)
    return builder.build_prompt(ref_code, backend, k)
