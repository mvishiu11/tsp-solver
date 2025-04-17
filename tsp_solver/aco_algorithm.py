import random
import math
from tsp_solver.base_algorithm import TSPAlgorithm
from tsp_solver.models import TSPProblem, TSPResult
from typing import Callable, Optional
import numpy as np


class TSPAntColony(TSPAlgorithm):
    """Ant Colony Optimization for TSP."""

    def __init__(
        self,
        n_ants=20,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        random_seed=42,
        iteration_callback: Optional[Callable[[int, list, float], None]] = None,
    ):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.random_seed = random_seed
        self.iteration_callback = iteration_callback
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def solve(self, problem: TSPProblem) -> TSPResult:
        coords = problem.city_coordinates
        n = len(coords)
        # Distance matrix
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = math.hypot(
                        coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]
                    )
                else:
                    dist[i, j] = 1e-9
        pheromone = np.ones((n, n))
        best_route = None
        best_distance = float("inf")
        for iter in range(self.n_iterations):
            all_routes = []
            all_distances = []
            for ant in range(self.n_ants):
                route = [random.randint(0, n - 1)]
                unvisited = set(range(n)) - {route[0]}
                while unvisited:
                    i = route[-1]
                    probs = []
                    for j in unvisited:
                        tau = pheromone[i][j] ** self.alpha
                        eta = (1.0 / dist[i][j]) ** self.beta
                        probs.append(tau * eta)
                    probs = np.array(probs)
                    probs /= probs.sum()
                    next_city = random.choices(list(unvisited), weights=probs)[0]
                    route.append(next_city)
                    unvisited.remove(next_city)
                all_routes.append(route)
                d = sum(dist[route[i], route[(i + 1) % n]] for i in range(n))
                all_distances.append(d)
                if d < best_distance:
                    best_route = list(route)
                    best_distance = d
            # Evaporate pheromone
            pheromone *= 1 - self.evaporation_rate
            # Deposit pheromone
            for route, d in zip(all_routes, all_distances):
                for i in range(n):
                    pheromone[route[i], route[(i + 1) % n]] += 1.0 / d
            if self.iteration_callback:
                self.iteration_callback(iter, best_route, best_distance)
        return TSPResult(best_route=best_route, best_distance=best_distance)
