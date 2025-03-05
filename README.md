# TSP Solver Comparison

A desktop application that solves the Traveling Salesman Problem (TSP) using two different approaches: a Genetic Algorithm (GA) and an exact dynamic programming algorithm. The application features a Tkinter-based GUI that visualizes the evolution of the GA solution alongside the exact solution, allowing for real-time comparisons and performance measurements.

## Features

- **Genetic Algorithm (GA) Solver**
  - Evolve TSP routes generation by generation.
  - Live, generation-by-generation updates on the GUI.
  - Adjustable GA parameters (population size, number of generations, mutation rate, tournament size).

- **Exact Solver**
  - Uses a dynamic programming approach (Held-Karp) to compute the optimal solution for small TSP instances.
  - Provides a reference for comparing the GA solution.

- **GUI Visualization**
  - Built with Tkinter and Matplotlib.
  - Side-by-side plots comparing the current GA solution with the exact solution.
  - Displays the execution time for each solver.
  - Allows users to modify GA parameters and TSP instance size between runs.

- **Modular & Extensible Design**
  - Object-oriented approach with a common abstract base class for TSP algorithms.
  - Uses Pydantic for well-defined, type-checked input and output models.
  - Designed for easy integration of additional algorithms or a more advanced GUI in the future.

- **Dependency Management**
  - Managed with Poetry for reproducibility and ease of installation.

## Project Structure

```
my-tsp-solver/
├── pyproject.toml         # Poetry configuration file
├── poetry.lock            # Locked dependency versions
├── README.md              # This file
├── tsp_solver
│   ├── __init__.py
│   ├── models.py           # Pydantic models for TSP input and output
│   ├── base_algorithm.py   # Abstract base class for TSP solvers
│   ├── ga_algorithm.py     # Genetic Algorithm implementation for TSP
│   ├── exact_algorithm.py  # Exact solver (Held-Karp) for TSP
│   ├── main.py             # Usage example for running the solvers
│   └── gui.py              # Tkinter-based GUI for solver visualization
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mvishiu11/tsp-solver.git
   cd tsp-solver
   ```

2. **Install Poetry (if not already installed):**

   ```bash
   pip install poetry
   ```

3. **Install the project dependencies:**

   ```bash
   poetry install
   sudo apt-get install python3-tk
   ```

## Running the Application

### GUI Mode

To launch the GUI application:

```bash
poetry run gui
```

The GUI window will open, where you can:
- Adjust the number of cities and GA parameters.
- Click the **Run** button to compute both the GA and exact solutions concurrently.
- View the GA solution evolving generation by generation alongside the exact solution.
- See the time taken for each solver.

### Command-Line Mode

You can run an example script that demonstrates how to use the TSP solvers programmatically:

```bash
poetry run example
```

## Linting

To run the lint script:

```bash
poetry run lint
```

This is also run as a pre-commit hook to ensure code consistency.

## Future Enhancements

- **Enhanced GUI:** Add more interactive controls, animations, and real-time parameter adjustments.
- **Additional Solvers:** Integrate alternative TSP solvers such as Simulated Annealing or Ant Colony Optimization.
- **Performance Improvements:** Optimize the algorithms for larger problem instances.
- **Extended Visualization:** Provide more detailed analytics and comparison metrics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions, improvements, or bug fixes.
