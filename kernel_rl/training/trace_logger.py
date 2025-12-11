"""
Async JSONL trace logger for training/debugging.

Provides a lightweight way to append per-step traces (prompts, outputs,
evaluations) to a run directory. Designed for concurrent async callers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

_trace_logger: "TraceLogger | None" = None


def set_trace_logger(logger: "TraceLogger | None") -> None:
    """Set the global trace logger instance."""
    global _trace_logger
    _trace_logger = logger


def get_trace_logger() -> "TraceLogger | None":
    """Get the global trace logger instance."""
    return _trace_logger


class TraceLogger:
    """Simple async JSONL logger with an internal lock."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, record: dict[str, Any]) -> None:
        """Append a JSON record to the trace file."""
        payload = json.dumps(record, default=str)
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(payload + "\n")
