# Prompt Format Reference

Use this file to map configuration → prompt shapes. Each section below shows the message layout for a specific configuration knob combination. File/line refs are given so you can trace the source templates quickly.

## Settings that affect prompt formatting
- `mode` (`single_turn` vs `multi_turn`): switches between one-shot prompts and the multi-turn refinement template.
- `reward_thinking_weight` (multi-turn only): toggles the think/no-think system prompt variant.
- `prompt_option` (`zero_shot`, `one_shot`, `few_shot`, `raicl`): chooses the user prompt template and whether retrieval snippets are injected.
- `renderer_name`: selects the renderer (e.g., `qwen3`, `llama3`), which controls chat wrappers and stop sequences.
- `backend` (`triton` or `cuda`): inserted into system prompts to steer code style and backend hints.
- `rag_index_path` / `raicl_k` (when `prompt_option=raicl`): include retrieved examples and how many make it into the prompt.
- `system_prompt` override (env-level hook): if provided, replaces the default system prompt for that env.

# Kevin No-Think Prompt Shape (RA-ICL Enabled)

Context: `thinking_weight=0` (Kevin “no-think”), but `prompt_option="raicl"` so the RA-ICL user payload still contains a `<think>...</think>` instruction even though the system prompt removes it. Renderer: qwen3 structured chat; messages alternate `[system]` then `[user]` per turn.

Sources:
- System prompt without think: `kernel_rl/envs/multiturn_kernelbench_env.py:MULTITURN_SYSTEM_PROMPT_NO_THINK`.
- RA-ICL user payload: `kernel_rl/rag/prompt_builder.py:RAICL_SYSTEM_PROMPT` and `RAICLPromptBuilder.build_prompt`.
- Message assembly (turn 0): `_build_initial_messages` in `multiturn_kernelbench_env.py`.
- Refinement assembly (turn >0): `_build_refinement_messages` in `multiturn_kernelbench_env.py`.

## Turn 0 (initial)
```
Inputs -> Model
────────────────────────────────────────────────────────
[system] MULTITURN_SYSTEM_PROMPT_NO_THINK
  - No <think> section; demands: respond with <KERNEL>…</KERNEL>.

[user] RA-ICL prompt (builder.build_prompt include_system=False)
  - "## Similar Optimization Examples" (k retrieved)
      • Example 1..k: PyTorch code + optimized {BACKEND} code
  - Problem block:
      • "Optimize the following PyTorch model..."
      • Explicit format rule: <think>…</think> then <KERNEL>…</KERNEL> (note the <think> ask survives)

Outputs (optimistic/intended for Kevin no-think)
────────────────────────────────────────────────────────
[assistant]
<KERNEL>
```python
class ModelNew(nn.Module):
    ...
```
</KERNEL>
```

## Turn 1 (first refinement)
```
Inputs -> Model
────────────────────────────────────────────────────────
[system] MULTITURN_SYSTEM_PROMPT_NO_THINK (same text)

[user]
  ├─ RA-ICL prompt (same as Turn 0; still contains <think> rule)
  └─ REFINEMENT_TEMPLATE (for prior Turn 0 attempt)
       • "## Previous Attempt (Turn 0)" + truncated kernel (≤2000 chars)
       • "## Evaluation Feedback"
           - Status, compiled?, tests passed/total, speedup (if any)
           - Error details block if present
       • "## Instructions" + guidance string
       • Reminder appended by env: "Remember: respond using <KERNEL>...</KERNEL>."

Outputs (optimistic/intended)
────────────────────────────────────────────────────────
[assistant]
<KERNEL>
```python
class ModelNew(nn.Module):
    # refined version
```
</KERNEL>
```

## Turn 2 (second refinement)
```
Inputs -> Model
────────────────────────────────────────────────────────
[system] same

[user]
  ├─ RA-ICL prompt (unchanged; still has <think> rule)
  └─ REFINEMENT_TEMPLATE for Turn 1 result
       • Previous Attempt (Turn 1) truncated kernel
       • Evaluation feedback for Turn 1
       • Guidance + "Remember: respond using <KERNEL>...</KERNEL>."

Outputs (optimistic/intended)
────────────────────────────────────────────────────────
[assistant]
<KERNEL>
```python
class ModelNew(nn.Module):
    # further refinement
```
</KERNEL>
```

## Turn 3 (third refinement)
```
Inputs -> Model
────────────────────────────────────────────────────────
[system] same

[user]
  ├─ RA-ICL prompt (unchanged; still has <think> rule)
  └─ REFINEMENT_TEMPLATE for Turn 2 result
       • Previous Attempt (Turn 2) truncated kernel
       • Evaluation feedback for Turn 2
       • Guidance + "Remember: respond using <KERNEL>...</KERNEL>."

Outputs (optimistic/intended)
────────────────────────────────────────────────────────
[assistant]
<KERNEL>
```python
class ModelNew(nn.Module):
    # third refinement
```
</KERNEL>
```

## Notes
- Even in no-think mode, every user payload embeds the RA-ICL format rule demanding `<think>...</think>`, so actual model outputs may include a think block unless explicitly stripped or post-processed.
- Stop sequences and tokenization are provided by the renderer; history stores `<think>` content for logging only and does not re-enter the prompt, but the RA-ICL template itself reintroduces the `<think>` instruction each turn.

# Multi-Turn With Thinking Enabled (mode=multi_turn, reward_thinking_weight>0)
Context: same env as Kevin but `_include_think=True`. System prompt is `MULTITURN_SYSTEM_PROMPT_WITH_THINK` (explicit <think> section). RA-ICL user payload still contains its own <think> rule.

Sources:
- System prompt with think: `multiturn_kernelbench_env.py:MULTITURN_SYSTEM_PROMPT_WITH_THINK`.
- RA-ICL user payload: `rag/prompt_builder.py`.
- Assembly: `_build_initial_messages` / `_build_refinement_messages`.

## Turn 0
```
Inputs -> Model
────────────────────────────────────────────────────────
[system] MULTITURN_SYSTEM_PROMPT_WITH_THINK
  - Requires <think>…</think> then <KERNEL>…</KERNEL>.

[user] RA-ICL prompt (include_system=False)
  - Retrieved examples + problem block
  - Again tells model: <think>…</think> then <KERNEL>…</KERNEL>.

Outputs (intended)
────────────────────────────────────────────────────────
[assistant]
<think>
- bullets ...
</think>
<KERNEL>
```python
class ModelNew(nn.Module):
    ...
```
</KERNEL>
```

## Turn >0
```
Inputs -> Model
────────────────────────────────────────────────────────
[system] MULTITURN_SYSTEM_PROMPT_WITH_THINK

[user]
  ├─ RA-ICL prompt (unchanged)
  └─ REFINEMENT_TEMPLATE for previous turn
       • Previous kernel (truncated) + eval feedback + guidance
       • Env reminder: "Remember: respond using <think>...</think> followed by <KERNEL>...</KERNEL>."

Outputs (intended)
────────────────────────────────────────────────────────
[assistant]
<think>…</think>
<KERNEL>…</KERNEL>
```

# Single-Turn Baseline (mode=single_turn)
Context: `KernelBenchEnv`. One round only. System prompt defaults to `DEFAULT_SYSTEM_PROMPT` (includes <think>). User prompt = `problem.prompt` (driven by `prompt_option`).

Sources:
- System prompt: `kernel_rl/envs/kernelbench_env.py:DEFAULT_SYSTEM_PROMPT`.
- User prompt selection: `kernel_rl/envs/kernelbench_client.py:get_prompt_for_problem`.

```
Inputs -> Model
────────────────────────────────────────────────────────
[system] DEFAULT_SYSTEM_PROMPT
  - Requires <think>…</think> then <KERNEL>…</KERNEL>.

[user] problem.prompt
  - If prompt_option=raicl: retrieved examples + problem + RA-ICL format rule (<think> then <KERNEL>)
  - Else: zero/one/few-shot template from prompt_constructor_toml (no RA-ICL block)

Outputs (intended)
────────────────────────────────────────────────────────
[assistant]
<think>…</think>
<KERNEL>
```python
class ModelNew(nn.Module):
    ...
```
</KERNEL>
```

# Prompt Option Variants (zero_shot / one_shot / few_shot / raicl)
Controls only the user payload; system prompt unchanged unless overridden.

Sources: `kernel_rl/envs/kernelbench_client.py:get_prompt_for_problem` and `rag/prompt_builder.py` for RA-ICL.

- `zero_shot`:
```
[user] Minimal problem statement + reference code; no examples.
```
- `one_shot`:
```
[user] Problem + single baked-in example from prompt_constructor_toml.
```
- `few_shot`:
```
[user] Problem + multiple canned examples (few-shot) from prompt_constructor_toml.
```
- `raicl`:
```
[user] Retrieved examples (k = raicl_k) + problem block + explicit format rule (<think>… then <KERNEL>…).
```

# Renderer Variants (renderer_name)
Renderers change the chat wrappers and stop sequences; message roles/content stay the same.

- `qwen3` (default): `<|im_start|>{role}\n{content}<|im_end|>` per message; stop token `<|im_end|>`. Strips assistant <think> from history by default; auto-prepends `<think>` if missing in assistant turns.
- `qwen3_disable_thinking`: Same, but prefill inserts `</think>` to suppress thinking tokens.
- `qwen3_instruct`: Same brackets, no auto-<think> and no thinking support.
- `deepseek_v3`: Inline `<|User|>` / `<|Assistant|>` tokens; assistant messages end with `<|end_of_sentence|>`. Disable-thinking variant prefixes `</think>`.
- `gpt_oss`: Harmony style `<|start|>role<|message|>...<|end|>`; optional baked system prompt if enabled.
- `role_colon` / `llama3`: Role:content or `[INST] ... [/INST]` style; stop sequences differ but payload ordering (system then user) is unchanged.

Sources: renderer implementations in `tinker_cookbook/renderers.py` (Qwen3, DeepSeekV3, GPT-OSS, RoleColon, Llama3).

# Backend Variants (triton vs cuda)
Only affects text substitution and example selection:
- System prompt `{backend}` placeholder becomes TRITON or CUDA.
- RA-ICL examples are filtered by backend and labeled “Optimized TRITON/CUDA Implementation”.
- Problem block reminders use the chosen backend name; message structure is unchanged.

# RA-ICL Retrieval Controls (rag_index_path, raicl_k)
Shape impact is confined to the user RA-ICL section:
- `rag_index_path` missing: retriever not loaded → fallback to `one_shot` prompt (no RA-ICL block).
- `raicl_k`: number of retrieved examples; scales the `## Similar Optimization Examples` block size.

```
[user] "## Similar Optimization Examples" (k entries) + problem block + format rule (<think>… then <KERNEL>…)
```

# Custom System Prompt Override
If `system_prompt` is passed to the env, it replaces the default/multi-turn system text; user payload is unchanged.

```
[system] <your custom text>
[user] (prompt_option content as configured)
```
Multi-turn still appends refinement text to the user block; only the system role content changes.
