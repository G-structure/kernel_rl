# cuTile Overview

cuTile is NVIDIA's tile-based parallel programming model and Python DSL for GPUs. It targets block-level work decomposition and automatically taps hardware like tensor cores and tensor memory accelerators while remaining portable across NVIDIA architectures.

## Execution Model
- Kernels are regular Python functions decorated with `@ct.kernel`; the host queues them with `ct.launch()` over a logical grid of blocks.
- Tile programs express block-level parallelism only: individual threads are abstracted away and cannot be directly addressed.
- No explicit intra-block synchronization is allowed; communication happens through global memory writes that other blocks can read.

## Data Model
- The model is array-centric—pointers are not exposed. Arrays live in global memory, are mutable, and can be passed from host to kernels.
- Tiles are immutable, block-local values created by loading from arrays or via factory ops; tile dimensions are compile-time powers of two and tiles are usable only inside tile code.
- Arrays can come from any DLPack or CUDA Array Interface provider (e.g., CuPy, PyTorch), and tile operations cover elementwise math, matmul, reductions, shape/transpose, and more.

## Typical Kernel Shape
1. Load one or more tiles from global arrays (`ct.load`).
2. Compute on the tiles (elementwise ops, matmul, reductions, etc.).
3. Store result tiles back to global arrays (`ct.store`).

## Requirements & Install
- GPUs with compute capability 10.x or 12.x, NVIDIA driver r580+, CUDA Toolkit 13.1+, and Python 3.10–3.13.
- Install via pip: `pip install cuda-tile`.
