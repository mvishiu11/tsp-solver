import random

from tsp_solver.ga_algorithm import TSPGAAlgorithm
from tsp_solver.models import TSPProblem


def main():
    # Example: generate random city coordinates
    random.seed(42)
    city_coords = [(random.random() * 100, random.random() * 100) for _ in range(20)]

    # Create a TSP problem
    problem = TSPProblem(city_coordinates=city_coords)

    # Create and run GA solver
    solver = TSPGAAlgorithm(population_size=100, generations=300, mutation_rate=0.1)
    result = solver.solve(problem)

    print("Best route found:", result.best_route)
    print("Best distance:", result.best_distance)


if __name__ == "__main__":
    main()
