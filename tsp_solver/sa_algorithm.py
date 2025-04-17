import random
import math
from tsp_solver.base_algorithm import TSPAlgorithm
from tsp_solver.models import TSPProblem, TSPResult
from typing import Callable, Optional


class TSPSimulatedAnnealing(TSPAlgorithm):
    """Simulated Annealing for TSP."""

    def __init__(
        self,
        max_iterations=1000,
        initial_temp=100.0,
        cooling_rate=0.995,
        random_seed=42,
        iteration_callback: Optional[Callable[[int, list, float], None]] = None,
    ):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.random_seed = random_seed
        self.iteration_callback = iteration_callback
        random.seed(self.random_seed)

    def solve(self, problem: TSPProblem) -> TSPResult:
        coords = problem.city_coordinates
        n = len(coords)

        def route_distance(route):
            return sum(
                math.hypot(
                    coords[route[i]][0] - coords[route[(i + 1) % n]][0],
                    coords[route[i]][1] - coords[route[(i + 1) % n]][1],
                )
                for i in range(n)
            )

        # Start with a random route
        current_route = list(range(n))
        random.shuffle(current_route)
        current_distance = route_distance(current_route)
        best_route = list(current_route)
        best_distance = current_distance
        temp = self.initial_temp
        for iter in range(self.max_iterations):
            # 2-opt swap: pick two cities and reverse the segment
            i, j = sorted(random.sample(range(n), 2))
            new_route = (
                current_route[:i]
                + current_route[i:j + 1][::-1]
                + current_route[j + 1:]
            )
            new_distance = route_distance(new_route)
            delta = new_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / (temp + 1e-9)):
                current_route = new_route
                current_distance = new_distance
                if current_distance < best_distance:
                    best_route = list(current_route)
                    best_distance = current_distance
            temp *= self.cooling_rate
            if self.iteration_callback:
                self.iteration_callback(iter, best_route, best_distance)
        return TSPResult(best_route=best_route, best_distance=best_distance)
