from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _default_runs_dir() -> Path:
    """Default location for run artifacts."""
    return Path.cwd() / "runs"


def create_app(runs_dir: Path | str | None = None) -> FastAPI:
    """
    Create the FastAPI app for the RL trace dashboard.

    Args:
        runs_dir: Directory containing training runs (each with traces.jsonl)
    """
    app = FastAPI(title="Kernel RL Dashboard", version="0.1.0")
    app.state.runs_dir = Path(runs_dir) if runs_dir else _default_runs_dir()

    # Simple CORS for local browsing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the dashboard UI."""
        index_path = static_dir / "index.html"
        return index_path.read_text(encoding="utf-8")

    @app.get("/api/runs")
    async def list_runs() -> JSONResponse:
        """List runs that contain traces.jsonl."""
        runs_root: Path = app.state.runs_dir
        if not runs_root.exists():
            return JSONResponse({"runs": []})

        runs = []
        for path in sorted(runs_root.iterdir()):
            if not path.is_dir():
                continue
            trace_path = path / "traces.jsonl"
            if not trace_path.exists():
                continue
            runs.append(
                {
                    "name": path.name,
                    "trace_path": str(trace_path),
                    "size_bytes": trace_path.stat().st_size,
                    "modified": trace_path.stat().st_mtime,
                }
            )
        return JSONResponse({"runs": runs})

    @app.get("/api/runs/{run_name}/traces")
    async def get_traces(
        run_name: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        mode: str | None = None,
        tail: bool = False,
    ) -> JSONResponse:
        """Return a slice of traces for a run."""
        run_path = Path(app.state.runs_dir) / run_name
        trace_path = run_path / "traces.jsonl"
        if not trace_path.exists():
            raise HTTPException(status_code=404, detail="traces.jsonl not found for this run")

        traces, total_records, actual_offset = _read_traces(
            trace_path, offset=offset, limit=limit, mode=mode, tail=tail
        )

        return JSONResponse(
            {
                "run": run_name,
                "offset": actual_offset,
                "limit": limit,
                "total": total_records,
                "mode": mode,
                "tail": tail,
                "traces": traces,
            }
        )

    @app.get("/api/runs/{run_name}/tb/tags")
    async def get_tb_tags(run_name: str) -> JSONResponse:
        """List scalar tags from TensorBoard events for a run."""
        logdir = _resolve_tb_dir(app.state.runs_dir, run_name)
        if logdir is None:
            raise HTTPException(status_code=404, detail="TensorBoard log dir not found")

        acc = _load_tb_accumulator(logdir)
        tags = sorted(acc.Tags().get("scalars", []))
        return JSONResponse({"run": run_name, "logdir": str(logdir), "tags": tags})

    @app.get("/api/runs/{run_name}/tb/summary")
    async def get_tb_summary(run_name: str) -> JSONResponse:
        """Return the latest value for each scalar tag."""
        logdir = _resolve_tb_dir(app.state.runs_dir, run_name)
        if logdir is None:
            raise HTTPException(status_code=404, detail="TensorBoard log dir not found")
        acc = _load_tb_accumulator(logdir)
        scalars = acc.Tags().get("scalars", [])
        latest: dict[str, Any] = {}
        for tag in scalars:
            vals = acc.Scalars(tag)
            if vals:
                last = vals[-1]
                latest[tag] = {"step": last.step, "value": last.value, "wall_time": last.wall_time}
        return JSONResponse({"run": run_name, "summary": latest})

    @app.get("/api/runs/{run_name}/config")
    async def get_run_config(run_name: str) -> JSONResponse:
        """Return config.json if present in the run directory."""
        run_path = Path(app.state.runs_dir) / run_name
        cfg_path = run_path / "config.json"
        if not cfg_path.exists():
            raise HTTPException(status_code=404, detail="config.json not found for this run")
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Failed to parse config.json: {exc}")
        return JSONResponse({"run": run_name, "config": data})

    @app.get("/api/runs/{run_name}/tb/scalars")
    async def get_tb_scalars(
        run_name: str,
        tag: str,
        limit: int = Query(500, ge=1, le=5000),
        tail: bool = True,
    ) -> JSONResponse:
        """Return scalar time series for a tag."""
        logdir = _resolve_tb_dir(app.state.runs_dir, run_name)
        if logdir is None:
            raise HTTPException(status_code=404, detail="TensorBoard log dir not found")

        acc = _load_tb_accumulator(logdir)
        scalars = acc.Scalars(tag)
        total = len(scalars)
        if tail:
            scalars = scalars[-limit:]
            start_idx = max(total - len(scalars), 0)
        else:
            scalars = scalars[:limit]
            start_idx = 0

        values = [
            {"wall_time": s.wall_time, "step": s.step, "value": s.value}
            for s in scalars
        ]
        return JSONResponse(
            {
                "run": run_name,
                "tag": tag,
                "total": total,
                "returned": len(values),
                "start_index": start_idx,
                "values": values,
            }
        )

    return app


def _read_traces(
    trace_path: Path,
    offset: int,
    limit: int,
    mode: str | None,
    tail: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Read traces from a JSONL file with optional filtering and tail support.

    Returns:
        traces: list of trace dicts
        total_records: number of records matching the filter
        actual_offset: offset used (for tail mode)
    """
    total_records = 0

    def _iter_filtered() -> Iterable[tuple[int, dict[str, Any]]]:
        with trace_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if mode and record.get("mode") != mode:
                    continue
                yield idx, record

    if tail:
        buffer: deque[tuple[int, dict[str, Any]]] = deque(maxlen=limit)
        for idx, record in _iter_filtered():
            total_records += 1
            buffer.append((idx, record))
        actual_offset = max(total_records - len(buffer), 0)
        traces = [_augment_record(idx, rec) for idx, rec in buffer]
        return traces, total_records, actual_offset

    traces: list[dict[str, Any]] = []
    for idx, record in _iter_filtered():
        if total_records >= offset and len(traces) < limit:
            traces.append(_augment_record(idx, record))
        total_records += 1
        if len(traces) >= limit:
            # Still count remaining records to report total
            continue
    return traces, total_records, offset


def _augment_record(idx: int, record: dict[str, Any]) -> dict[str, Any]:
    """Attach display-friendly metadata to a record."""
    out = dict(record)
    out.setdefault("meta", {})
    out["meta"]["line_index"] = idx
    out["meta"]["summary"] = _summarize_record(record)
    return out


def _summarize_record(record: dict[str, Any]) -> str:
    """Build a short summary string for list cards."""
    level = record.get("level")
    pid = record.get("problem_id")
    mode = record.get("mode", "?")
    reward = record.get("reward")
    reward_str = f"{reward:.3f}" if isinstance(reward, (int, float)) else "?"
    turn = record.get("turn")
    return f"L{level} P{pid} T{turn} | {mode} | r={reward_str}"


def _resolve_tb_dir(runs_dir: Path, run_name: str) -> Path | None:
    """Find a TensorBoard log directory for the run."""
    run_path = Path(runs_dir) / run_name
    tb_dir = run_path / "tensorboard"
    if tb_dir.exists():
        return tb_dir
    # fallback: look for events* in run root
    for cand in run_path.glob("events.*"):
        return run_path
    return None


def _load_tb_accumulator(logdir: Path) -> EventAccumulator:
    """Load a TensorBoard event accumulator."""
    if not logdir.exists():
        raise HTTPException(status_code=404, detail="TensorBoard logdir missing")

    has_events = any(logdir.glob("events.*")) if logdir.is_dir() else logdir.name.startswith("events.")
    if not has_events:
        # If logdir is a directory, EventAccumulator will still work, but give a clearer message.
        raise HTTPException(status_code=404, detail="No TensorBoard event files found")

    acc = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel RL trace dashboard")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(_default_runs_dir()),
        help="Directory containing run folders (each with traces.jsonl)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8009, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload (dev only)")
    args = parser.parse_args()

    app = create_app(args.runs_dir)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
