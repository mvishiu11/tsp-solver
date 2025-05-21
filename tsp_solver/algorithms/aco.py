# algorithms/aco.py
from __future__ import annotations
import random
import math
import time
from typing import List
import numpy as np

from tsp_solver.models import TSPProblem, TSPResult
from tsp_solver.algorithms.base import TSPAlgorithm
from tsp_solver.utils.events import update_queue, cancel_event, pause_event


class TSPAntColony(TSPAlgorithm):
    """Ant Colony Optimization with default parameters α=1, β=2, evap=0.5."""

    def __init__(
        self,
        n_ants: int = 20,
        n_iter: int = 200,
        alpha: float = 1.0,
        beta: float = 2.0,
        evap: float = 0.5,
        random_seed: int | None = None,
    ):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.evap = evap
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def solve(self, problem: TSPProblem) -> TSPResult:
        coords = problem.city_coordinates
        n = len(coords)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                D[i, j] = D[j, i] = d or 1e-9
        tau = np.ones((n, n))

        best_route: List[int] | None = None
        best_dist = float("inf")
        start = time.time()

        for it in range(self.n_iter):
            if cancel_event.is_set():
                break
            pause_event.wait()

            routes, dists = [], []
            for _ in range(self.n_ants):
                route = [random.randrange(n)]
                unvisited = set(range(n)) - {route[0]}
                while unvisited:
                    i = route[-1]
                    probs = []
                    for j in unvisited:
                        probs.append((tau[i, j] ** self.alpha) * ((1 / D[i, j]) ** self.beta))
                    probs = np.array(probs)
                    probs /= probs.sum()
                    nxt = random.choices(list(unvisited), probs)[0]
                    route.append(nxt)
                    unvisited.remove(nxt)
                dist = sum(D[route[k], route[(k + 1) % n]] for k in range(n))
                routes.append(route)
                dists.append(dist)
                if dist < best_dist:
                    best_route, best_dist = route, dist

            # pheromone update
            tau *= 1 - self.evap
            for r, d in zip(routes, dists):
                for k in range(n):
                    tau[r[k], r[(k + 1) % n]] += 1.0 / d

            update_queue.put(
                dict(
                    algo_key="aco",
                    coords=coords,
                    route=best_route,
                    distance=best_dist,
                    runtime=time.time() - start,
                    iteration=it,
                )
            )

        return TSPResult(best_route=best_route, best_distance=best_dist)
