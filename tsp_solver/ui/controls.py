# ui/controls.py
"""Top-bar controls (parameters + buttons) with pause/resume/step/stop hooks."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

from ttkbootstrap import ttk

_LBL_CFG = dict(padx=2, pady=3, sticky="w")
_ENTRY_W = 8


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
        # two-column grid (label | widget) for slim vertical sidebar
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        row = 0
        ttk.Label(self, text="City Count:").grid(row=row, column=0, **_LBL_CFG)
        self.city_count = ttk.Entry(self, width=_ENTRY_W)
        self.city_count.insert(0, "15")
        self.city_count.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="Population:").grid(row=row, column=0, **_LBL_CFG)
        self.pop_size = ttk.Entry(self, width=_ENTRY_W)
        self.pop_size.insert(0, "80")
        self.pop_size.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="Generations:").grid(row=row, column=0, **_LBL_CFG)
        self.generations = ttk.Entry(self, width=_ENTRY_W)
        self.generations.insert(0, "300")
        self.generations.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="Mutation Rate:").grid(row=row, column=0, **_LBL_CFG)
        self.mutation_rate = ttk.Entry(self, width=_ENTRY_W)
        self.mutation_rate.insert(0, "0.10")
        self.mutation_rate.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="Algorithm:").grid(row=row, column=0, **_LBL_CFG)
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
        self.algo_combo.grid(row=row, column=1, padx=2, pady=3)

        # ---------- GA hyperparameters ----------
        row += 1
        ttk.Label(self, text="GA Tournament K:").grid(row=row, column=0, **_LBL_CFG)
        self.tournament_k = ttk.Entry(self, width=_ENTRY_W)
        self.tournament_k.insert(0, "3")
        self.tournament_k.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="GA Max No Improve:").grid(row=row, column=0, **_LBL_CFG)
        self.ga_max_no_improve = ttk.Entry(self, width=_ENTRY_W)
        self.ga_max_no_improve.insert(0, "50")
        self.ga_max_no_improve.grid(row=row, column=1, padx=2, pady=3)

        # ---------- SA hyperparameters ----------
        row += 1
        ttk.Label(self, text="SA Max Iterations:").grid(row=row, column=0, **_LBL_CFG)
        self.sa_max_iterations = ttk.Entry(self, width=_ENTRY_W)
        self.sa_max_iterations.insert(0, "1500")
        self.sa_max_iterations.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="SA Initial Temp:").grid(row=row, column=0, **_LBL_CFG)
        self.sa_initial_temp = ttk.Entry(self, width=_ENTRY_W)
        self.sa_initial_temp.insert(0, "100.0")
        self.sa_initial_temp.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="SA Cooling:").grid(row=row, column=0, **_LBL_CFG)
        self.sa_cooling = ttk.Entry(self, width=_ENTRY_W)
        self.sa_cooling.insert(0, "0.995")
        self.sa_cooling.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="SA Min Temp:").grid(row=row, column=0, **_LBL_CFG)
        self.sa_min_temp = ttk.Entry(self, width=_ENTRY_W)
        self.sa_min_temp.insert(0, "0.001")
        self.sa_min_temp.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="SA Max No Improve:").grid(row=row, column=0, **_LBL_CFG)
        self.sa_max_no_improve = ttk.Entry(self, width=_ENTRY_W)
        self.sa_max_no_improve.insert(0, "100")
        self.sa_max_no_improve.grid(row=row, column=1, padx=2, pady=3)

        # ---------- ACO hyperparameters ----------
        row += 1
        ttk.Label(self, text="ACO Ants:").grid(row=row, column=0, **_LBL_CFG)
        self.aco_n_ants = ttk.Entry(self, width=_ENTRY_W)
        self.aco_n_ants.insert(0, "20")
        self.aco_n_ants.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="ACO Iterations:").grid(row=row, column=0, **_LBL_CFG)
        self.aco_n_iter = ttk.Entry(self, width=_ENTRY_W)
        self.aco_n_iter.insert(0, "200")
        self.aco_n_iter.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="ACO Alpha:").grid(row=row, column=0, **_LBL_CFG)
        self.aco_alpha = ttk.Entry(self, width=_ENTRY_W)
        self.aco_alpha.insert(0, "1.0")
        self.aco_alpha.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="ACO Beta:").grid(row=row, column=0, **_LBL_CFG)
        self.aco_beta = ttk.Entry(self, width=_ENTRY_W)
        self.aco_beta.insert(0, "2.0")
        self.aco_beta.grid(row=row, column=1, padx=2, pady=3)

        row += 1
        ttk.Label(self, text="ACO Evap:").grid(row=row, column=0, **_LBL_CFG)
        self.aco_evap = ttk.Entry(self, width=_ENTRY_W)
        self.aco_evap.insert(0, "0.5")
        self.aco_evap.grid(row=row, column=1, padx=2, pady=3)

        row += 2  # small spacer before buttons

        self.run_btn = ttk.Button(self, text="Run", command=on_run, bootstyle="success")
        self.run_btn.grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        row += 1
        self.pause_btn = ttk.Button(
            self, text="Pause", command=on_pause, state="disabled", bootstyle="warning"
        )
        self.pause_btn.grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2
        )

        row += 1
        self.resume_btn = ttk.Button(
            self, text="Resume", command=on_resume, state="disabled", bootstyle="info"
        )
        self.resume_btn.grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2
        )

        row += 1
        self.step_btn = ttk.Button(
            self, text="Step", command=on_step, state="disabled", bootstyle="secondary"
        )
        self.step_btn.grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        row += 1
        self.stop_btn = ttk.Button(
            self, text="Stop", command=on_stop, state="disabled", bootstyle="danger"
        )
        self.stop_btn.grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
