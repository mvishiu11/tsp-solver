# ui/main.py   –  drop-in replacement
"""
Material-styled TSP visualiser.
• Nothing runs until the user presses Run.
• If “Exact” is selected with >18 cities the user is warned first.
• Queue is purged on startup so no stale messages appear.
"""
from __future__ import annotations

import random
import signal
import sys
import threading
from typing import Dict, Any, List

import queue
import tkinter as tk
from ttkbootstrap import ttk

from tsp_solver.ui.theme import create_root
from tsp_solver.ui.controls import ControlBar
from tsp_solver.ui.plot_area import PlotArea
from tsp_solver.models import TSPProblem
from tsp_solver.algorithms.ga import TSPGAAlgorithm
from tsp_solver.algorithms.sa import TSPSimulatedAnnealing
from tsp_solver.algorithms.aco import TSPAntColony
from tsp_solver.algorithms.exact import TSPExactAlgorithm
from tsp_solver.utils.events import update_queue, cancel_event, pause_event
from tsp_solver.utils.threading_utils import start_solver_thread

# ------------------------------------------------------------------ #
_threads: List[threading.Thread] = []        # active workers
# ------------------------------------------------------------------ #


def main() -> None:
    # guarantee an empty queue when the GUI appears
    while not update_queue.empty():
        try:
            update_queue.get_nowait()
        except queue.Empty:
            break

    root = create_root()
    root.protocol("WM_DELETE_WINDOW", lambda: _on_stop(root))

    plot = PlotArea(root)
    plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    ctrl = ControlBar(
        root,
        on_run=lambda: _on_run(ctrl, plot),
        on_pause=lambda: _on_pause(ctrl),
        on_resume=lambda: _on_resume(ctrl),
        on_step=lambda: _on_step(ctrl, root),
        on_stop=lambda: _on_stop(root),
    )
    ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    credits = ttk.Label(
        root,
        text="Created by Jakub Muszyński, Mateusz Nędzi, Patryk Zan",
        font=("Segoe UI", 9, "italic"),
        foreground="#888",
    )
    credits.pack(side=tk.TOP, pady=(0, 8))

    # ---------------- poll one message per tick ---------------- #
    def poll():
        try:
            msg: Dict[str, Any] = update_queue.get_nowait()
            plot.update_plot(**msg)
        except queue.Empty:
            pass

        # re-enable Run when all threads finished
        if _threads and all(not t.is_alive() for t in _threads):
            ctrl.run_btn.configure(state="normal")
            ctrl.pause_btn.configure(state="disabled")
            ctrl.resume_btn.configure(state="disabled")
            ctrl.step_btn.configure(state="disabled")
            ctrl.stop_btn.configure(state="disabled")
            _threads.clear()

        root.after(50, poll)        # ~20 fps

    poll()

    # allow Ctrl-C
    signal.signal(signal.SIGINT, lambda *_: _on_stop(root))
    try:
        root.mainloop()
    except KeyboardInterrupt:
        _on_stop(root)


# ------------------------------------------------------------------ #
# button callbacks
# ------------------------------------------------------------------ #
def _on_run(ctrl: ControlBar, plot: PlotArea) -> None:
    if cancel_event.is_set():
        cancel_event.clear()

    plot.reset_all_titles()

    # read parameters
    try:
        n_cities = int(ctrl.city_count.get())
        pop_size = int(ctrl.pop_size.get())
        generations = int(ctrl.generations.get())
        mut_rate = float(ctrl.mutation_rate.get())
    except ValueError:
        ttk.messagebox.showerror("Invalid input", "Numeric parameters required.")
        return

    # warn if Exact on too many cities
    selection = ctrl.algo_var.get()
    if selection in ("All", "Exact") and n_cities > 18:
        if not ttk.messagebox.askyesno(
            "Exact solver warning",
            (
                "The Held–Karp exact solver is exponential and may run for hours "
                f"with {n_cities} cities.\n\n"
                "Continue anyway?"
            ),
        ):
            return

    # create problem instance
    random.seed(42)
    coords = [(random.random() * 100, random.random() * 100) for _ in range(n_cities)]
    problem = TSPProblem(city_coordinates=coords)

    # helper to spawn each solver
    def spawn(name: str, solver_cls):
        thread = start_solver_thread(
            name,
            lambda: solver_cls().solve(problem)
            if name != "GA"
            else solver_cls(
                population_size=pop_size,
                generations=generations,
                mutation_rate=mut_rate,
            ).solve(problem),
        )
        _threads.append(thread)

    if selection in ("All", "Genetic Algorithm"):
        spawn("GA", TSPGAAlgorithm)
    if selection in ("All", "Exact"):
        spawn("Exact", TSPExactAlgorithm)
    if selection in ("All", "Simulated Annealing"):
        spawn("SA", TSPSimulatedAnnealing)
    if selection in ("All", "Ant Colony"):
        spawn("ACO", TSPAntColony)

    # button states
    ctrl.run_btn.configure(state="disabled")
    ctrl.pause_btn.configure(state="normal")
    ctrl.resume_btn.configure(state="disabled")
    ctrl.step_btn.configure(state="normal")
    ctrl.stop_btn.configure(state="normal")
    pause_event.set()          # ensure running


def _on_pause(ctrl: ControlBar) -> None:
    pause_event.clear()
    ctrl.pause_btn.configure(state="disabled")
    ctrl.resume_btn.configure(state="normal")
    ctrl.step_btn.configure(state="normal")


def _on_resume(ctrl: ControlBar) -> None:
    pause_event.set()
    ctrl.pause_btn.configure(state="normal")
    ctrl.resume_btn.configure(state="disabled")
    ctrl.step_btn.configure(state="disabled")


def _on_step(ctrl: ControlBar, root) -> None:
    if pause_event.is_set():  # only when paused
        return
    pause_event.set()
    root.after(60, pause_event.clear)   # let exactly one loop body run


def _on_stop(root) -> None:
    cancel_event.set()
    pause_event.set()
    for t in _threads:
        t.join(timeout=0.2)
    root.destroy()
    sys.exit(0)


if __name__ == "__main__":
    main()
