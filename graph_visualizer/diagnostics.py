"""Runtime diagnostics for graph visualizer crashes and long load phases."""

from __future__ import annotations

from datetime import datetime
import faulthandler
from pathlib import Path
import sys
import traceback
from typing import Any

_LOG_HANDLE: Any | None = None
_LOG_PATH: Path | None = None


def install_crash_diagnostics(log_path: str | Path | None = None) -> Path:
    """Enable best-effort logging for Python errors and native fatal crashes."""
    global _LOG_HANDLE, _LOG_PATH
    if _LOG_HANDLE is not None and _LOG_PATH is not None:
        return _LOG_PATH
    path = Path(log_path) if log_path is not None else Path(__file__).resolve().parents[1] / "graph_visualizer_crash.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_HANDLE = path.open("a", encoding="utf-8", buffering=1)
    _LOG_PATH = path
    log_event("crash diagnostics enabled", path=str(path))
    try:
        faulthandler.enable(file=_LOG_HANDLE, all_threads=True)
    except Exception as exc:
        log_event("faulthandler enable failed", error=repr(exc))

    previous_hook = sys.excepthook

    def _hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        log_exception("uncaught exception", exc=exc, tb=tb)
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _hook
    return path


def log_event(message: str, **fields: Any) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    suffix = ""
    if fields:
        suffix = " " + " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
    line = f"[{timestamp}] {message}{suffix}"
    print(line, flush=True)
    if _LOG_HANDLE is not None:
        try:
            _LOG_HANDLE.write(line + "\n")
            _LOG_HANDLE.flush()
        except Exception:
            pass


def log_exception(message: str, exc: BaseException, tb: Any | None = None, **fields: Any) -> None:
    log_event(message, error=repr(exc), **fields)
    if _LOG_HANDLE is None:
        return
    try:
        traceback.print_exception(type(exc), exc, tb if tb is not None else exc.__traceback__, file=_LOG_HANDLE)
        _LOG_HANDLE.flush()
    except Exception:
        pass


def _format_value(value: Any) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) > 240:
        return repr(text[:237] + "...")
    return repr(text)
