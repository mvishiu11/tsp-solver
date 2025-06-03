# ui/main.py   –  drop-in replacement
"""
Material-styled TSP visualiser.
• Nothing runs until the user presses Run.
• If "Exact" is selected with >18 cities the user is warned first.
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
_threads: List[threading.Thread] = []  # active workers
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

    # -------- layout: plot on left, controls on right -------- #
    content = ttk.Frame(root)
    content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Plot area (left)
    plot = PlotArea(content)
    plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

    # Controls + credits (right)
    right_pane = ttk.Frame(content)
    right_pane.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)

    ctrl = ControlBar(
        right_pane,
        on_run=lambda: _on_run(ctrl, plot),
        on_pause=lambda: _on_pause(ctrl),
        on_resume=lambda: _on_resume(ctrl),
        on_step=lambda: _on_step(ctrl, root),
        on_stop=lambda: _on_stop(root),
    )
    ctrl.pack(side=tk.TOP, fill=tk.Y, padx=0, pady=0)

    credits = ttk.Label(
        right_pane,
        text="Created by Jakub Muszyński, Mateusz Nędzi, Patryk Zan",
        font=("Segoe UI", 9, "italic"),
        foreground="#888",
    )
    credits.pack(side=tk.TOP, pady=(10, 0))

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

        root.after(50, poll)  # ~20 fps

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

        # GA extras
        tournament_k = int(ctrl.tournament_k.get())
        ga_max_no_improve = int(ctrl.ga_max_no_improve.get())

        # SA parameters
        sa_max_iterations = int(ctrl.sa_max_iterations.get())
        sa_initial_temp = float(ctrl.sa_initial_temp.get())
        sa_cooling = float(ctrl.sa_cooling.get())
        sa_min_temp = float(ctrl.sa_min_temp.get())
        sa_max_no_improve = int(ctrl.sa_max_no_improve.get())

        # ACO parameters
        aco_n_ants = int(ctrl.aco_n_ants.get())
        aco_n_iter = int(ctrl.aco_n_iter.get())
        aco_alpha = float(ctrl.aco_alpha.get())
        aco_beta = float(ctrl.aco_beta.get())
        aco_evap = float(ctrl.aco_evap.get())
    except ValueError:
        ttk.messagebox.showerror("Invalid input", "Numeric parameters required.")
        return

    # warn if Exact on too many cities
    selection = ctrl.algo_var.get()
    skip_exact = False
    if "Exact" in (selection,):
        if n_cities > 18:
            skip_exact = True
        else:
            skip_exact = False

    print(skip_exact)

    # create problem instance
    random.seed(42)
    coords = [(random.random() * 100, random.random() * 100) for _ in range(n_cities)]
    problem = TSPProblem(city_coordinates=coords)

    # sequential runner function
    def run_algorithms_sequentially():
        import time

        algorithms_to_run = []

        # build list of algorithms to run
        if selection in ("All", "Genetic Algorithm"):
            algorithms_to_run.append(
                (
                    "ga",
                    TSPGAAlgorithm,
                    dict(
                        population_size=pop_size,
                        generations=generations,
                        mutation_rate=mut_rate,
                        tournament_k=tournament_k,
                        max_no_improve=ga_max_no_improve,
                    ),
                )
            )

        if not skip_exact and selection in ("All", "Exact"):
            algorithms_to_run.append(("exact", TSPExactAlgorithm, {}))

        if selection in ("All", "Simulated Annealing"):
            algorithms_to_run.append(
                (
                    "sa",
                    TSPSimulatedAnnealing,
                    dict(
                        max_iterations=sa_max_iterations,
                        initial_temp=sa_initial_temp,
                        cooling=sa_cooling,
                        min_temp=sa_min_temp,
                        max_no_improve=sa_max_no_improve,
                    ),
                )
            )

        if selection in ("All", "Ant Colony"):
            algorithms_to_run.append(
                (
                    "aco",
                    TSPAntColony,
                    dict(
                        n_ants=aco_n_ants,
                        n_iter=aco_n_iter,
                        alpha=aco_alpha,
                        beta=aco_beta,
                        evap=aco_evap,
                    ),
                )
            )

        # run each algorithm sequentially
        for algo_key, solver_cls, kwargs in algorithms_to_run:
            if cancel_event.is_set():
                break

            # send start message so UI begins timing
            update_queue.put(
                dict(
                    algo_key=algo_key,
                    coords=coords,
                    route=[],
                    distance=float("inf"),
                    runtime=None,
                    iteration="starting",
                )
            )

            start_time = time.perf_counter()
            solver = solver_cls(**kwargs)
            result = solver.solve(problem)
            end_time = time.perf_counter()

            # send final result without runtime so UI calculates real elapsed time
            update_queue.put(
                dict(
                    algo_key=algo_key,
                    coords=coords,
                    route=result.best_route,
                    distance=result.best_distance,
                    runtime=None,  # let UI calculate real elapsed time
                    iteration="final",
                )
            )

    # start sequential execution in background thread
    thread = start_solver_thread("SequentialRunner", run_algorithms_sequentially)
    _threads.append(thread)

    # button states
    ctrl.run_btn.configure(state="disabled")
    ctrl.pause_btn.configure(state="normal")
    ctrl.resume_btn.configure(state="disabled")
    ctrl.step_btn.configure(state="normal")
    ctrl.stop_btn.configure(state="normal")
    pause_event.set()  # ensure running


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
    root.after(60, pause_event.clear)  # let exactly one loop body run


def _on_stop(root) -> None:
    cancel_event.set()
    pause_event.set()
    for t in _threads:
        t.join(timeout=0.2)
    root.destroy()
    sys.exit(0)


if __name__ == "__main__":
    main()
