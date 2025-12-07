#!/usr/bin/env python3
"""
Build RAG index from KernelBook and AI-CUDA-Engineer-Archive datasets.

Usage:
    python -m kernel_rl.scripts.build_rag_index --output ./kernel_index

    # Triton only (KernelBook)
    python -m kernel_rl.scripts.build_rag_index --output ./kernel_index --triton-only

    # CUDA only (Sakana)
    python -m kernel_rl.scripts.build_rag_index --output ./kernel_index --cuda-only
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index for kernel retrieval",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./kernel_rag_index",
        help="Output directory for the index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/codebert-base",
        help="Embedding model (sentence-transformers compatible)",
    )
    parser.add_argument(
        "--triton-only",
        action="store_true",
        help="Only include KernelBook (Triton) examples",
    )
    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="Only include Sakana (CUDA) examples",
    )
    parser.add_argument(
        "--sakana-levels",
        type=str,
        default="1,2,3",
        help="Comma-separated Sakana levels to include (default: 1,2,3)",
    )
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include incorrect kernels from Sakana (not recommended)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embedding model (cuda/cpu/None for auto)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for datasets",
    )

    args = parser.parse_args()

    # Parse sakana levels
    sakana_levels = [int(x) for x in args.sakana_levels.split(",")]

    # Determine what to include
    include_kernelbook = not args.cuda_only
    include_sakana = not args.triton_only

    if args.triton_only and args.cuda_only:
        logger.error("Cannot specify both --triton-only and --cuda-only")
        sys.exit(1)

    logger.info("Loading corpus...")
    logger.info(f"  KernelBook (Triton): {include_kernelbook}")
    logger.info(f"  Sakana (CUDA): {include_sakana}")
    if include_sakana:
        logger.info(f"  Sakana levels: {sakana_levels}")
        logger.info(f"  Correct only: {not args.include_incorrect}")

    from kernel_rl.rag.corpus import KernelCorpus
    from kernel_rl.rag.retriever import KernelRetriever

    # Load corpus
    corpus = KernelCorpus(
        include_kernelbook=include_kernelbook,
        include_sakana=include_sakana,
        sakana_levels=sakana_levels,
        sakana_correct_only=not args.include_incorrect,
    )
    corpus.load(cache_dir=args.cache_dir)

    if len(corpus) == 0:
        logger.error("No examples loaded! Check dataset availability.")
        sys.exit(1)

    # Build index
    logger.info(f"Building index with {args.model}...")
    retriever = KernelRetriever(model_name=args.model, device=args.device)
    retriever.build_index(corpus, batch_size=args.batch_size)

    # Save
    logger.info(f"Saving index to {args.output}...")
    retriever.save(args.output)

    logger.info("Done!")

    # Print stats
    triton_count = len(corpus.filter_by_backend("triton"))
    cuda_count = len(corpus.filter_by_backend("cuda"))
    logger.info(f"Index stats:")
    logger.info(f"  Total examples: {len(corpus)}")
    logger.info(f"  Triton examples: {triton_count}")
    logger.info(f"  CUDA examples: {cuda_count}")


if __name__ == "__main__":
    main()
