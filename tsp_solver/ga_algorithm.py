import random
from typing import Callable, List, Optional

import numpy as np

from tsp_solver.base_algorithm import TSPAlgorithm
from tsp_solver.models import TSPProblem, TSPResult


class TSPGAAlgorithm(TSPAlgorithm):
    """Genetic Algorithm for TSP."""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        mutation_rate: float = 0.1,
        tournament_k: int = 3,
        random_seed: int = 42,
        generation_callback: Optional[Callable[[int, List[int], float], None]] = None,
    ):
        """
        :param population_size: Number of individuals in each generation
        :param generations: Number of generations to evolve
        :param mutation_rate: Probability of mutation
        :param tournament_k: Tournament size for selection
        :param random_seed: Seed for reproducibility
        :param generation_callback: A function called each generation with signature:
                                   callback(generation, best_route, best_distance).
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.random_seed = random_seed
        self.generation_callback = generation_callback

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def solve(self, problem: TSPProblem) -> TSPResult:
        """Solve the TSP using a Genetic Algorithm and return the best route found."""
        city_coords = problem.city_coordinates
        num_cities = len(city_coords)

        # Precompute distance matrix (optional, for speed)
        distance_matrix = self._compute_distance_matrix(city_coords)

        # 1. Initialize population
        population = self._create_initial_population(num_cities)

        best_route = None
        best_distance = float("inf")

        for gen in range(self.generations):
            # 2. Evaluate fitness of each individual
            fitnesses = [
                self._route_distance(ind, distance_matrix) for ind in population
            ]

            # Track best solution in this generation
            gen_best_idx = np.argmin(fitnesses)
            gen_best_distance = fitnesses[gen_best_idx]
            if gen_best_distance < best_distance:
                best_distance = gen_best_distance
                best_route = population[gen_best_idx][:]

            # 3. Create next generation
            new_population = []
            while len(new_population) < self.population_size:
                # 3a. Selection
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)

                # 3b. Crossover
                child1 = self._order_crossover(parent1, parent2)
                child2 = self._order_crossover(parent2, parent1)

                # 3c. Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._swap_mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._swap_mutation(child2)

                new_population.extend([child1, child2])

            # ---- call the callback to update the GUI ----
            if self.generation_callback:
                self.generation_callback(gen, best_route, best_distance)

            population = new_population

            # (Optional) Print progress
            if gen % 50 == 0:
                print(f"Generation {gen}: Best Distance = {best_distance}")

        return TSPResult(best_route=best_route, best_distance=best_distance)

    # ---------- Helper Methods ---------- #

    def _compute_distance_matrix(self, city_coords: List[tuple]) -> np.ndarray:
        """Compute and return a distance matrix for the given city coordinates."""
        num_cities = len(city_coords)
        distance_matrix = np.zeros((num_cities, num_cities))

        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    dx = city_coords[i][0] - city_coords[j][0]
                    dy = city_coords[i][1] - city_coords[j][1]
                    distance_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

        return distance_matrix

    def _create_initial_population(self, num_cities: int) -> List[List[int]]:
        """Create an initial population of random permutations."""
        population = []
        for _ in range(self.population_size):
            route = list(range(num_cities))
            random.shuffle(route)
            population.append(route)
        return population

    def _route_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate the total distance of the route."""
        dist = 0.0
        for i in range(len(route) - 1):
            dist += distance_matrix[route[i], route[i + 1]]
        # Return to start
        dist += distance_matrix[route[-1], route[0]]
        return dist

    def _tournament_selection(
        self, population: List[List[int]], fitnesses: List[float]
    ) -> List[int]:
        """Select a single parent using tournament selection."""
        selected_indices = random.sample(range(len(population)), self.tournament_k)
        best_index = min(selected_indices, key=lambda idx: fitnesses[idx])
        return population[best_index][:]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX)."""
        size = len(parent1)
        child = [-1] * size

        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]

        # Fill in from parent2
        pointer = end
        for city in parent2:
            if city not in child:
                if pointer >= size:
                    pointer = 0
                child[pointer] = city
                pointer += 1

        return child

    def _swap_mutation(self, route: List[int]) -> List[int]:
        """Swap two cities in the route."""
        a, b = random.sample(range(len(route)), 2)
        route[a], route[b] = route[b], route[a]
        return route
