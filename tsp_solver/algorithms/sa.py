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
        min_temp: float = 1e-3,
        max_no_improve: int = 100,
    ) -> None:
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling = cooling
        self.min_temp = min_temp
        self.max_no_improve = max_no_improve

        if random_seed is not None:
            random.seed(random_seed)

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
        start = time.perf_counter()
        paused_time = 0.0  # accumulated pause duration
        no_improve = 0

        for it in range(self.max_iterations):
            if cancel_event.is_set():
                break

            # handle pause – exclude paused duration from runtime
            if not pause_event.is_set():
                _pause_start = time.perf_counter()
                pause_event.wait()
                paused_time += time.perf_counter() - _pause_start

            # propose a 2-opt swap
            i, j = sorted(random.sample(range(n), 2))
            candidate = current[:i] + current[i : j + 1][::-1] + current[j + 1 :]
            cand_d = route_distance(candidate)
            delta = cand_d - current_d

            # accept or reject
            if delta < 0 or random.random() < math.exp(-delta / (T + 1e-9)):
                current, current_d = candidate, cand_d
                if current_d < best_d:
                    best, best_d = current[:], current_d
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            # cool down
            T *= self.cooling

            # send progress (no runtime yet)
            update_queue.put(
                {
                    "algo_key": "sa",
                    "coords": coords,
                    "route": best,
                    "distance": best_d,
                    "runtime": time.perf_counter() - start - paused_time,
                    "iteration": it,
                }
            )

            # early stop?
            if T < self.min_temp or no_improve >= self.max_no_improve:
                break

        total_time = time.perf_counter() - start - paused_time
        update_queue.put(
            {
                "algo_key": "sa",
                "coords": coords,
                "route": best,
                "distance": best_d,
                "runtime": total_time,
                "iteration": it,
            }
        )

        return TSPResult(best_route=best, best_distance=best_d)
