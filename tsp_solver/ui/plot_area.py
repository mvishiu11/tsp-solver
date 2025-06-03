# ui/plot_area.py
"""Fast, flicker-free Matplotlib panel for four algorithms."""
from __future__ import annotations

import matplotlib
import time

import tkinter as tk
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from ttkbootstrap import ttk

matplotlib.use("TkAgg")

_ALGO_TITLES = {
    "ga": "Genetic Algorithm",
    "exact": "Exact (Held-Karp)",
    "sa": "Simulated Annealing",
    "aco": "Ant Colony",
}


class PlotArea(ttk.Frame):
    """
    2×2 grid of subplots.  Each subplot keeps:
        • a scatter of city nodes   (drawn once)
        • a Line2D object for route (updated every iteration)
    This avoids full clears → no flicker, visible real-time progress.
    """

    def __init__(self, master) -> None:
        super().__init__(master)

        # slightly smaller figure width and reduced padding so plots sit closer
        self.figure, self.axes = plt.subplots(2, 2, figsize=(10, 7))
        self.figure.tight_layout(pad=1.5)

        # maps algo_key ➜ (row, col) index & the live Line2D object
        self._ax_map: Dict[str, Tuple[int, int]] = {
            "ga": (0, 0),
            "exact": (0, 1),
            "sa": (1, 0),
            "aco": (1, 1),
        }
        self._lines: Dict[str, plt.Line2D] = {}
        self._scatter_drawn: Dict[str, bool] = {k: False for k in _ALGO_TITLES}
        # map algo_key ➜ first-arrival timestamp (perf_counter)
        self._start_times: Dict[str, float] = {}

        for key, (r, c) in self._ax_map.items():
            ax = self.axes[r, c]
            ax.set_title(_ALGO_TITLES[key])
            ax.set_aspect("equal", anchor="C")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle=":", alpha=0.4)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        NavigationToolbar2Tk(self.canvas, self).pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #
    def update_plot(
        self,
        algo_key: str,
        coords: List[Tuple[float, float]],
        route: List[int],
        distance: float,
        runtime: float | None,
        iteration: int | str,
    ) -> None:
        """
        Modify the existing line/scatter for *algo_key* in-place.
        Executed on the Tk main thread (called from queue-poll loop).
        """
        ax = self.axes[self._ax_map[algo_key]]

        # draw city scatter once
        if not self._scatter_drawn[algo_key] and coords:
            xs, ys = zip(*coords)
            ax.scatter(xs, ys, c="#1f77b4", s=30, edgecolors="k", zorder=3)
            for i, (x, y) in enumerate(coords):
                ax.text(
                    x,
                    y,
                    str(i),
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.1"),
                )
            self._scatter_drawn[algo_key] = True

        # update / create the tour line
        if coords and route:
            xs = [coords[i][0] for i in route] + [coords[route[0]][0]]
            ys = [coords[i][1] for i in route] + [coords[route[0]][1]]

            if algo_key not in self._lines:
                (line,) = ax.plot(
                    xs,
                    ys,
                    linewidth=2,
                    color="#00e676",
                    alpha=0.85,
                    zorder=2,
                )
                self._lines[algo_key] = line
            else:
                line = self._lines[algo_key]
                line.set_data(xs, ys)

        # title with metrics
        title = f"{_ALGO_TITLES[algo_key]}   "
        if isinstance(iteration, int):
            title += f"Iter {iteration}   "

        # always calculate real elapsed time (ignore algorithm's provided runtime)
        if algo_key not in self._start_times:
            self._start_times[algo_key] = time.perf_counter()
        elapsed = time.perf_counter() - self._start_times[algo_key]

        title += f"Dist={distance:.2f}   Time={elapsed:.2f}s"
        ax.set_title(title)

        self.canvas.draw_idle()

    # --------------------------------------------------------------------- #
    # UTILS
    # --------------------------------------------------------------------- #
    def reset_all_titles(self) -> None:
        """Clear titles & remove previous route lines for a fresh run."""
        # reset runtime origins as well
        self._start_times.clear()

        for key, (r, c) in self._ax_map.items():
            ax = self.axes[r, c]
            ax.set_title(_ALGO_TITLES[key])
            if key in self._lines:
                self._lines[key].remove()
                del self._lines[key]
            self._scatter_drawn[key] = False

        self.canvas.draw_idle()

    def reset_selected_algorithms(self, algo_keys: list[str]) -> None:
        """Clear titles & remove route lines only for specified algorithms."""
        # reset runtime origins for selected algorithms
        for key in algo_keys:
            if key in self._start_times:
                del self._start_times[key]

        for key in algo_keys:
            if key in self._ax_map:
                r, c = self._ax_map[key]
                ax = self.axes[r, c]
                ax.set_title(_ALGO_TITLES[key])
                if key in self._lines:
                    self._lines[key].remove()
                    del self._lines[key]
                # Note: don't reset _scatter_drawn[key] since cities should stay

        self.canvas.draw_idle()
