# utils/threading_utils.py
"""Helpers to launch solver threads with clean shutdown."""

from __future__ import annotations
import threading
from typing import Callable, Any, Tuple


def start_solver_thread(
    name: str,
    target: Callable[..., Any],
    /,
    *args: Tuple[Any, ...],
    daemon: bool = True,
    **kwargs,
) -> threading.Thread:
    """
    Start *target* in a background thread.

    Parameters
    ----------
    name : str            Thread.name (helpful in debuggers)
    target : Callable     Function to run
    *args, **kwargs       Forwarded to *target*
    daemon : bool         Default True so app exits cleanly
    """
    thread = threading.Thread(
        name=name, target=target, args=args, kwargs=kwargs, daemon=daemon
    )
    thread.start()
    return thread
