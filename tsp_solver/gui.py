import random
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

from tsp_solver.exact_algorithm import TSPExactAlgorithm
from tsp_solver.ga_algorithm import TSPGAAlgorithm
from tsp_solver.models import TSPProblem

matplotlib.use("TkAgg")


class TSPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSP Solver Comparison")
        # Responsive window size: 80% of screen
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        w, h = int(screen_w * 0.8), int(screen_h * 0.8)
        self.geometry(f"{w}x{h}")
        self.minsize(800, 600)
        # Allow resizing
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        # Apply modern ttk theme
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 11))
        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Removed SIGINT signal handler for reliability

        # ---------------------------
        # Input Controls Frame (Multi-row Responsive)
        # ---------------------------
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        for i in range(16):
            control_frame.grid_columnconfigure(i, weight=1)

        # Row 0: City/genetic parameters
        ttk.Label(control_frame, text="City Count:").grid(
            row=0, column=0, padx=2, pady=3, sticky="e"
        )
        self.city_count_entry = ttk.Entry(control_frame, width=7)
        self.city_count_entry.insert(0, "10")
        self.city_count_entry.grid(row=0, column=1, padx=2, pady=3)

        ttk.Label(control_frame, text="Population Size:").grid(
            row=0, column=2, padx=2, pady=3, sticky="e"
        )
        self.pop_size_entry = ttk.Entry(control_frame, width=7)
        self.pop_size_entry.insert(0, "50")
        self.pop_size_entry.grid(row=0, column=3, padx=2, pady=3)

        ttk.Label(control_frame, text="Generations:").grid(
            row=0, column=4, padx=2, pady=3, sticky="e"
        )
        self.generations_entry = ttk.Entry(control_frame, width=7)
        self.generations_entry.insert(0, "200")
        self.generations_entry.grid(row=0, column=5, padx=2, pady=3)

        ttk.Label(control_frame, text="Mutation Rate:").grid(
            row=0, column=6, padx=2, pady=3, sticky="e"
        )
        self.mutation_rate_entry = ttk.Entry(control_frame, width=7)
        self.mutation_rate_entry.insert(0, "0.1")
        self.mutation_rate_entry.grid(row=0, column=7, padx=2, pady=3)

        ttk.Label(control_frame, text="Tournament Size:").grid(
            row=0, column=8, padx=2, pady=3, sticky="e"
        )
        self.tournament_entry = ttk.Entry(control_frame, width=7)
        self.tournament_entry.insert(0, "3")
        self.tournament_entry.grid(row=0, column=9, padx=2, pady=3)

        # Row 1: Algorithm selection, sliders, run/pause/resume/step
        ttk.Label(control_frame, text="Algorithm:").grid(
            row=1, column=0, padx=2, pady=3, sticky="e"
        )
        self.algorithm_var = tk.StringVar()
        self.algorithm_combo = ttk.Combobox(
            control_frame,
            textvariable=self.algorithm_var,
            values=["Genetic Algorithm", "Exact", "Simulated Annealing", "Ant Colony"],
            state="readonly",
            width=18,
        )
        self.algorithm_combo.current(0)
        self.algorithm_combo.grid(row=1, column=1, padx=2, pady=3)

        ttk.Label(control_frame, text="Population Size (slider)").grid(
            row=1, column=2, padx=2, pady=3, sticky="e"
        )
        self.pop_size_slider = ttk.Scale(
            control_frame,
            from_=10,
            to=200,
            value=50,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.on_pop_size_slider,
        )
        self.pop_size_slider.grid(row=1, column=3, padx=2, pady=3)
        self.pop_size_value_label = ttk.Label(control_frame, text="50")
        self.pop_size_value_label.grid(row=1, column=4, padx=2, pady=3)

        ttk.Label(control_frame, text="Mutation Rate (slider)").grid(
            row=1, column=5, padx=2, pady=3, sticky="e"
        )
        self.mutation_slider = ttk.Scale(
            control_frame,
            from_=0.01,
            to=0.5,
            value=0.1,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.on_mutation_slider,
        )
        self.mutation_slider.grid(row=1, column=6, padx=2, pady=3)
        self.mutation_value_label = ttk.Label(control_frame, text="0.10")
        self.mutation_value_label.grid(row=1, column=7, padx=2, pady=3)

        self.run_button = ttk.Button(
            control_frame, text="Run", command=self.start_solvers
        )
        self.run_button.grid(row=1, column=8, padx=2, pady=3)
        self.pause_button = ttk.Button(
            control_frame, text="Pause", command=self.pause_solver, state=tk.DISABLED
        )
        self.pause_button.grid(row=1, column=9, padx=2, pady=3)
        self.resume_button = ttk.Button(
            control_frame, text="Resume", command=self.resume_solver, state=tk.DISABLED
        )
        self.resume_button.grid(row=1, column=10, padx=2, pady=3)
        self.step_button = ttk.Button(
            control_frame, text="Step", command=self.step_solver, state=tk.DISABLED
        )
        self.step_button.grid(row=1, column=11, padx=2, pady=3)

        # ---------------------------
        # Matplotlib Figure for Plots (4 subplots)
        # ---------------------------
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.ax_ga = self.axes[0, 0]
        self.ax_exact = self.axes[0, 1]
        self.ax_sa = self.axes[1, 0]
        self.ax_aco = self.axes[1, 1]
        self.fig.suptitle("TSP: GA | Exact | SA | ACO")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10
        )

        # Store city coordinates, run times
        self.city_coords = []
        self.ga_run_time = None
        self.exact_run_time = None
        self.sa_run_time = None
        self.aco_run_time = None

        # Threads
        self.ga_thread = None
        self.exact_thread = None
        self.sa_thread = None
        self.aco_thread = None

    def start_solvers(self):
        """Triggered when the user clicks 'Run'. Reads parameters and starts all 4 solvers concurrently."""
        self.run_button.config(state=tk.DISABLED)
        try:
            city_count = int(self.city_count_entry.get())
            pop_size = int(self.pop_size_entry.get())
            generations = int(self.generations_entry.get())
            mutation_rate = float(self.mutation_rate_entry.get())
            tournament_k = int(self.tournament_entry.get())
        except ValueError:
            tk.messagebox.showerror(
                "Invalid Input", "Please enter valid numeric parameters."
            )
            self.run_button.config(state=tk.NORMAL)
            return
        random.seed(42)
        self.city_coords = [
            (random.random() * 100, random.random() * 100) for _ in range(city_count)
        ]
        # Clear previous plots
        for ax, title in zip(
            [self.ax_ga, self.ax_exact, self.ax_sa, self.ax_aco],
            ["Genetic Algorithm", "Exact", "Simulated Annealing", "Ant Colony"],
        ):
            ax.clear()
            ax.set_title(title)
        self.fig.suptitle("TSP: GA | Exact | SA | ACO")
        self.canvas.draw()
        problem = TSPProblem(city_coordinates=self.city_coords)
        # Start all 4 solvers in separate threads
        self.ga_thread = threading.Thread(
            target=self.run_ga,
            args=(problem, pop_size, generations, mutation_rate, tournament_k),
            daemon=True,
        )
        self.exact_thread = threading.Thread(
            target=self.run_exact, args=(problem,), daemon=True
        )
        self.sa_thread = threading.Thread(
            target=self.run_sa, args=(problem,), daemon=True
        )
        self.aco_thread = threading.Thread(
            target=self.run_aco, args=(problem,), daemon=True
        )
        self.ga_thread.start()
        self.exact_thread.start()
        self.sa_thread.start()
        self.aco_thread.start()
        self.after(100, self.check_threads)

    def check_threads(self):
        """Check if all 4 threads have finished."""
        threads = [self.ga_thread, self.exact_thread, self.sa_thread, self.aco_thread]
        if all(t is None or not t.is_alive() for t in threads):
            self.run_button.config(state=tk.NORMAL)
        else:
            self.after(100, self.check_threads)

    def on_mutation_slider(self, val):
        self.mutation_value_label.config(text=f"{float(val):.2f}")
        print(f"Mutation rate slider changed: {val}")

    def on_pop_size_slider(self, val):
        self.pop_size_value_label.config(text=f"{int(float(val))}")
        print(f"Population size slider changed: {val}")

    def pause_solver(self):
        print("Pause pressed (not yet implemented)")

    def resume_solver(self):
        print("Resume pressed (not yet implemented)")

    def step_solver(self):
        print("Step pressed (not yet implemented)")

    def run_sa(self, problem):
        from tsp_solver.sa_algorithm import TSPSimulatedAnnealing

        def callback(iter, route, dist):
            self._draw_route(self.ax_sa, route, f"SA Iter {iter}")
            self.ax_sa.set_title(f"Simulated Annealing\nIter {iter}, Best: {dist:.2f}")
            self.canvas.draw_idle()

        sa = TSPSimulatedAnnealing(max_iterations=500, iteration_callback=callback)
        result = sa.solve(problem)
        self._draw_route(self.ax_sa, result.best_route, "SA Final")
        self.ax_sa.set_title(f"Simulated Annealing\nBest: {result.best_distance:.2f}")
        self.canvas.draw_idle()

    def run_aco(self, problem):
        from tsp_solver.aco_algorithm import TSPAntColony

        def callback(iter, route, dist):
            self._draw_route(self.ax_aco, route, f"ACO Iter {iter}")
            self.ax_aco.set_title(f"Ant Colony\nIter {iter}, Best: {dist:.2f}")
            self.canvas.draw_idle()

        aco = TSPAntColony(n_ants=20, n_iterations=200, iteration_callback=callback)
        result = aco.solve(problem)
        self._draw_route(self.ax_aco, result.best_route, "ACO Final")
        self.ax_aco.set_title(f"Ant Colony\nBest: {result.best_distance:.2f}")
        self.canvas.draw_idle()

    def run_ga(self, problem, pop_size, generations, mutation_rate, tournament_k):
        """Run the GA solver and update the GA subplot on each generation."""
        start_time = time.time()

        def on_generation(gen, best_route, best_distance):
            """Callback for GA progress; schedule UI update in main thread."""
            self.after(0, self.update_ga_plot, gen, best_route, best_distance)

        solver = TSPGAAlgorithm(
            population_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            tournament_k=tournament_k,
            random_seed=123,
            generation_callback=on_generation,
        )
        result = solver.solve(problem)
        self.ga_run_time = time.time() - start_time

        # Final update for GA plot with run time info
        self.after(
            0, self.update_ga_plot, "Final", result.best_route, result.best_distance
        )

    def run_exact(self, problem):
        """Run the Exact solver and update the Exact subplot when done."""
        start_time = time.time()
        solver = TSPExactAlgorithm()
        result = solver.solve(problem)
        self.exact_run_time = time.time() - start_time

        self.after(0, self.update_exact_plot, result.best_route, result.best_distance)

    def update_ga_plot(self, generation, route, distance):
        """Update the GA subplot with current best route and info."""
        self.ax_ga.clear()
        title = f"GA - Gen {generation} | Dist={distance:.2f}"
        if self.ga_run_time is not None:
            title += f" | Time: {self.ga_run_time:.2f}s"
        self.ax_ga.set_title(title)
        self._draw_route(self.ax_ga, route, "GA Route")
        self.canvas.draw()

    def update_exact_plot(self, route, distance):
        """Update the Exact subplot with final route and info."""
        self.ax_exact.clear()
        title = f"Exact | Dist={distance:.2f} | Time: {self.exact_run_time:.2f}s"
        self.ax_exact.set_title(title)
        self._draw_route(self.ax_exact, route, "Exact Route")
        self.canvas.draw()

    def _draw_route(self, ax, route, label, top_routes=None):
        """Polished plot of the route using NetworkX. Optionally overlays top_routes."""
        if not route or not self.city_coords:
            return
        coords = self.city_coords
        G = nx.Graph()
        for idx, (x, y) in enumerate(coords):
            G.add_node(idx, pos=(x, y))
        # Add edges for the main route
        edges = [(route[i], route[(i + 1) % len(route)]) for i in range(len(route))]
        G.add_edges_from(edges)
        # Layout: circular for <=10, spring for more
        if len(coords) <= 10:
            pos = nx.circular_layout(G, scale=40)
        else:
            pos = nx.spring_layout(G, scale=40, seed=42)
        # Rescale and center to fit nicely
        xs = [xy[0] for xy in pos.values()]
        ys = [xy[1] for xy in pos.values()]
        margin = 10
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        for k in pos:
            x, y = pos[k]
            pos[k] = (
                (x - min_x) / (max_x - min_x + 1e-9) * 80 + margin,
                (y - min_y) / (max_y - min_y + 1e-9) * 60 + margin,
            )
        # Draw subtle background
        ax.set_facecolor("#f8f9fa")
        # Draw all nodes
        node_size = max(250, 1000 // max(1, len(coords)))
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            node_color="#1f77b4",
            edgecolors="k",
            linewidths=1.5,
            alpha=0.92,
        )
        # Draw all edges (light gray for context)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=1, alpha=0.35)
        # Draw top-k routes (if provided)
        if top_routes:
            for i, r in enumerate(top_routes):
                e = [(r[j], r[(j + 1) % len(r)]) for j in range(len(r))]
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=e,
                    ax=ax,
                    edge_color="#ff7f0e",
                    width=2,
                    alpha=0.15 + 0.12 * i,
                )
        # Draw main route (best) with glow
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax, edge_color="#00e676", width=7, alpha=0.65
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax, edge_color="#00796b", width=2, alpha=0.95
        )
        # Draw labels
        for i, (x, y) in pos.items():
            ax.text(
                x,
                y,
                str(i),
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"),
            )

    def on_closing(self):
        # Only destroy if the window exists
        try:
            if self.winfo_exists():
                self.destroy()
        except Exception:
            pass


def main():
    app = TSPApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.destroy()
    sys.exit(0)


if __name__ == "__main__":
    main()
