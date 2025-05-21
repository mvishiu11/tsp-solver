"""Centralised theme / style helpers based on ttkbootstrap."""
from __future__ import annotations

import ttkbootstrap as ttk

_THEMENAME = "litera"


def create_root(title: str = "Travelling Salesman Visualizer") -> ttk.Window:
    """
    Return a ttkbootstrap root window with a material-style theme applied.
    """
    root = ttk.Window(themename=_THEMENAME)
    root.title(title)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = int(sw * 0.8), int(sh * 0.8)
    root.geometry(f"{w}x{h}")
    root.minsize(800, 600)
    return root
