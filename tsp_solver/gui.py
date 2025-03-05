import random
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tsp_solver.exact_algorithm import TSPExactAlgorithm
from tsp_solver.ga_algorithm import TSPGAAlgorithm
from tsp_solver.models import TSPProblem

matplotlib.use("TkAgg")


class TSPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSP Solver Comparison")
        self.geometry("1200x700")

        # ---------------------------
        # Input Controls Frame
        # ---------------------------
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # City Count
        ttk.Label(control_frame, text="City Count:").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        self.city_count_entry = ttk.Entry(control_frame, width=10)
        self.city_count_entry.insert(0, "10")
        self.city_count_entry.grid(row=0, column=1, padx=5, pady=5)

        # GA: Population Size
        ttk.Label(control_frame, text="Population Size:").grid(
            row=0, column=2, padx=5, pady=5, sticky="e"
        )
        self.pop_size_entry = ttk.Entry(control_frame, width=10)
        self.pop_size_entry.insert(0, "50")
        self.pop_size_entry.grid(row=0, column=3, padx=5, pady=5)

        # GA: Generations
        ttk.Label(control_frame, text="Generations:").grid(
            row=0, column=4, padx=5, pady=5, sticky="e"
        )
        self.generations_entry = ttk.Entry(control_frame, width=10)
        self.generations_entry.insert(0, "200")
        self.generations_entry.grid(row=0, column=5, padx=5, pady=5)

        # GA: Mutation Rate
        ttk.Label(control_frame, text="Mutation Rate:").grid(
            row=0, column=6, padx=5, pady=5, sticky="e"
        )
        self.mutation_rate_entry = ttk.Entry(control_frame, width=10)
        self.mutation_rate_entry.insert(0, "0.1")
        self.mutation_rate_entry.grid(row=0, column=7, padx=5, pady=5)

        # GA: Tournament Size
        ttk.Label(control_frame, text="Tournament Size:").grid(
            row=0, column=8, padx=5, pady=5, sticky="e"
        )
        self.tournament_entry = ttk.Entry(control_frame, width=10)
        self.tournament_entry.insert(0, "3")
        self.tournament_entry.grid(row=0, column=9, padx=5, pady=5)

        # Run Button
        self.run_button = ttk.Button(
            control_frame, text="Run", command=self.start_solvers
        )
        self.run_button.grid(row=0, column=10, padx=10, pady=5)

        # ---------------------------
        # Matplotlib Figure for Plots
        # ---------------------------
        self.fig, (self.ax_ga, self.ax_exact) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.suptitle("TSP: GA (left) vs Exact (right)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Store city coordinates, GA and exact run times
        self.city_coords = []
        self.ga_run_time = None
        self.exact_run_time = None

        # Threads
        self.ga_thread = None
        self.exact_thread = None

    def start_solvers(self):
        """Triggered when the user clicks 'Run'. Reads parameters and starts GA and Exact solvers concurrently."""
        # Disable run button while processing
        self.run_button.config(state=tk.DISABLED)

        # Read parameters from entries
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

        # Generate random city coordinates (for reproducibility, we use a random seed)
        random.seed(42)
        self.city_coords = [
            (random.random() * 100, random.random() * 100) for _ in range(city_count)
        ]

        # Clear previous plots
        self.ax_ga.clear()
        self.ax_exact.clear()
        self.fig.suptitle("TSP: GA (left) vs Exact (right)")
        self.canvas.draw()

        # Create TSP problem instance
        problem = TSPProblem(city_coordinates=self.city_coords)

        # Start GA and Exact solvers in separate threads
        self.ga_thread = threading.Thread(
            target=self.run_ga,
            args=(problem, pop_size, generations, mutation_rate, tournament_k),
            daemon=True,
        )
        self.exact_thread = threading.Thread(
            target=self.run_exact, args=(problem,), daemon=True
        )

        self.ga_thread.start()
        self.exact_thread.start()

        # Monitor threads to re-enable the run button once both are done
        self.after(100, self.check_threads)

    def check_threads(self):
        """Check if both GA and Exact threads have finished."""
        if (self.ga_thread is None or not self.ga_thread.is_alive()) and (
            self.exact_thread is None or not self.exact_thread.is_alive()
        ):
            self.run_button.config(state=tk.NORMAL)
        else:
            self.after(100, self.check_threads)

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

    def _draw_route(self, ax, route, label):
        """Plot the route on the given Axes object."""
        if not route:
            return
        coords = [self.city_coords[i] for i in route]
        # Append the start city at the end to close the loop
        coords.append(self.city_coords[route[0]])
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        ax.plot(xs, ys, marker="o", linestyle="-", label=label)
        # Optionally annotate cities with their index
        for idx, (x, y) in enumerate(self.city_coords):
            ax.text(x, y, str(idx), fontsize=8, color="blue")
        ax.legend()

    def on_closing(self):
        """Called when the window is closed."""
        self.quit()
        self.destroy()
        sys.exit(0)


def main():
    try:
        app = TSPApp()
        app.mainloop()
    except KeyboardInterrupt:
        app.destroy


if __name__ == "__main__":
    main()
