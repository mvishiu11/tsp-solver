# algorithms/sa.py
from __future__ import annotations

import math
import random
import time
from typing import List

from tsp_solver.models import TSPProblem, TSPResult

from tsp_solver.algorithms.base import TSPAlgorithm
from tsp_solver.utils.events import update_queue, cancel_event, pause_event


class TSPSimulatedAnnealing(TSPAlgorithm):
    """2-opt Simulated Annealing with exponential cooling."""

    def __init__(
        self,
        max_iterations: int = 1500,
        initial_temp: float = 100.0,
        cooling: float = 0.995,
        random_seed: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling = cooling
        if random_seed is not None:
            random.seed(random_seed)

    # ---------- core ----------
    def solve(self, problem: TSPProblem) -> TSPResult:
        coords = problem.city_coordinates
        n = len(coords)

        def route_distance(route: List[int]) -> float:
            return sum(
                math.hypot(
                    coords[route[i]][0] - coords[route[(i + 1) % n]][0],
                    coords[route[i]][1] - coords[route[(i + 1) % n]][1],
                )
                for i in range(n)
            )

        # initial random tour
        current = random.sample(range(n), n)
        current_d = route_distance(current)
        best, best_d = current[:], current_d

        T = self.initial_temp
        start = time.time()

        for it in range(self.max_iterations):
            if cancel_event.is_set():
                break
            pause_event.wait()

            i, j = sorted(random.sample(range(n), 2))
            candidate = current[:i] + current[i: j + 1][::-1] + current[j + 1:]
            cand_d = route_distance(candidate)
            delta = cand_d - current_d

            if delta < 0 or random.random() < math.exp(-delta / (T + 1e-9)):
                current, current_d = candidate, cand_d
                if current_d < best_d:
                    best, best_d = current[:], current_d

            T *= self.cooling

            update_queue.put(
                dict(
                    algo_key="sa",
                    coords=coords,
                    route=best,
                    distance=best_d,
                    runtime=time.time() - start,
                    iteration=it,
                )
            )

        return TSPResult(best_route=best, best_distance=best_d)
