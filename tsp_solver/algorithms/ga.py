from __future__ import annotations

import random
import time

import numpy as np
from tsp_solver.models import TSPProblem, TSPResult
from tsp_solver.algorithms.base import TSPAlgorithm
from tsp_solver.utils.events import update_queue, cancel_event, pause_event


class TSPGAAlgorithm(TSPAlgorithm):
    """
    Genetic Algorithm for TSP with order-crossover and swap-mutation.
    Emits a queue message every generation so the UI can redraw.
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 400,
        mutation_rate: float = 0.10,
        tournament_k: int = 3,
        max_no_improve: int = 50,
        random_seed: int | None = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.max_no_improve = max_no_improve

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    # ---------- TSPAlgorithm API ---------- #
    def solve(self, problem: TSPProblem) -> TSPResult:
        coords = problem.city_coordinates
        n = len(coords)
        dist_mtx = self._dist_matrix(coords)
        pop = [random.sample(range(n), n) for _ in range(self.population_size)]

        best_route: list[int] | None = None
        best_dist = float("inf")
        no_improve = 0
        start = time.perf_counter()

        for gen in range(self.generations):
            if cancel_event.is_set():
                break
            pause_event.wait()

            # evaluate fitness
            fitness = [self._route_dist(ind, dist_mtx) for ind in pop]

            # check for new global best
            gbest_idx = int(np.argmin(fitness))
            gbest_route = pop[gbest_idx]
            gbest_dist = fitness[gbest_idx]
            if gbest_dist < best_dist:
                best_dist, best_route = gbest_dist, gbest_route
                no_improve = 0
            else:
                no_improve += 1

            # send progress update
            update_queue.put({
                "algo_key": "ga",
                "coords": coords,
                "route": best_route,
                "distance": best_dist,
                "runtime": time.perf_counter() - start,
                "iteration": gen,
            })

            # early stop if no improvement for too long
            if no_improve >= self.max_no_improve:
                break

            # produce next generation
            new_pop: list[list[int]] = []
            while len(new_pop) < self.population_size:
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)
                c1 = self._order_crossover(p1, p2)
                c2 = self._order_crossover(p2, p1)
                if random.random() < self.mutation_rate:
                    self._swap_mut(c1)
                if random.random() < self.mutation_rate:
                    self._swap_mut(c2)
                new_pop.extend([c1, c2])
            pop = new_pop[: self.population_size]

        return TSPResult(best_route=best_route, best_distance=best_dist)

    # ---------- helpers ---------- #
    @staticmethod
    def _dist_matrix(coords):
        n = len(coords)
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(
                    coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]
                )
                m[i, j] = m[j, i] = d
        return m

    @staticmethod
    def _route_dist(route: list[int], m) -> float:
        return sum(m[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))

    def _tournament(self, pop, fit):
        idxs = random.sample(range(len(pop)), self.tournament_k)
        return pop[min(idxs, key=lambda i: fit[i])][:]

    @staticmethod
    def _order_crossover(p1: list[int], p2: list[int]) -> list[int]:
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b] = p1[a:b]
        fill = [c for c in p2 if c not in child]
        ptr = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[ptr]
                ptr += 1
        return child

    @staticmethod
    def _swap_mut(route: list[int]) -> None:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
