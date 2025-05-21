# ui/controls.py
"""Top-bar controls (parameters + buttons) with pause/resume/step/stop hooks."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

from ttkbootstrap import ttk

_LBL_CFG = dict(padx=2, pady=3, sticky="e")
_ENTRY_W = 7


class ControlBar(ttk.Frame):
    """Numeric inputs + algorithm selector + Run / Pause / Resume / Step / Stop."""

    def __init__(
        self,
        master,
        *,
        on_run: Callable[[], None],
        on_pause: Callable[[], None],
        on_resume: Callable[[], None],
        on_step: Callable[[], None],
        on_stop: Callable[[], None],
    ):
        super().__init__(master)
        self.columnconfigure(tuple(range(16)), weight=1)

        # ---------- numeric parameters ----------
        ttk.Label(self, text="City Count:").grid(row=0, column=0, **_LBL_CFG)
        self.city_count = ttk.Entry(self, width=_ENTRY_W)
        self.city_count.insert(0, "25")
        self.city_count.grid(row=0, column=1, padx=2, pady=3)

        ttk.Label(self, text="Population:").grid(row=0, column=2, **_LBL_CFG)
        self.pop_size = ttk.Entry(self, width=_ENTRY_W)
        self.pop_size.insert(0, "80")
        self.pop_size.grid(row=0, column=3, padx=2, pady=3)

        ttk.Label(self, text="Generations:").grid(row=0, column=4, **_LBL_CFG)
        self.generations = ttk.Entry(self, width=_ENTRY_W)
        self.generations.insert(0, "300")
        self.generations.grid(row=0, column=5, padx=2, pady=3)

        ttk.Label(self, text="Mutation Rate:").grid(row=0, column=6, **_LBL_CFG)
        self.mutation_rate = ttk.Entry(self, width=_ENTRY_W)
        self.mutation_rate.insert(0, "0.10")
        self.mutation_rate.grid(row=0, column=7, padx=2, pady=3)

        # ---------- algorithm selector ----------
        ttk.Label(self, text="Algorithm:").grid(row=1, column=0, **_LBL_CFG)
        self.algo_var = tk.StringVar(self)
        self.algo_combo = ttk.Combobox(
            self,
            textvariable=self.algo_var,
            values=[
                "All",
                "Genetic Algorithm",
                "Exact",
                "Simulated Annealing",
                "Ant Colony",
            ],
            state="readonly",
            width=20,
        )
        self.algo_combo.current(0)
        self.algo_combo.grid(row=1, column=1, padx=2, pady=3)

        # ---------- action buttons ----------
        self.run_btn = ttk.Button(self, text="Run", command=on_run, bootstyle="success")
        self.run_btn.grid(row=1, column=8, padx=2, pady=3)

        self.pause_btn = ttk.Button(
            self, text="Pause", command=on_pause, state="disabled", bootstyle="warning"
        )
        self.pause_btn.grid(row=1, column=9, padx=2, pady=3)

        self.resume_btn = ttk.Button(
            self, text="Resume", command=on_resume, state="disabled", bootstyle="info"
        )
        self.resume_btn.grid(row=1, column=10, padx=2, pady=3)

        self.step_btn = ttk.Button(
            self, text="Step", command=on_step, state="disabled", bootstyle="secondary"
        )
        self.step_btn.grid(row=1, column=11, padx=2, pady=3)

        self.stop_btn = ttk.Button(
            self, text="Stop", command=on_stop, state="disabled", bootstyle="danger"
        )
        self.stop_btn.grid(row=1, column=12, padx=2, pady=3)
