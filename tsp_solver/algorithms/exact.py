# algorithms/exact.py
"""
Exact Held–Karp solver (O(n²·2ⁿ)) with cooperative
Pause / Stop and a final queue update for the UI.
"""
from __future__ import annotations

import math
import time
from typing import List, Tuple

from tsp_solver.models import TSPProblem, TSPResult

from tsp_solver.algorithms.base import TSPAlgorithm
from tsp_solver.utils.events import update_queue, cancel_event, pause_event


class TSPExactAlgorithm(TSPAlgorithm):
    """
    Dynamic-programming Held–Karp.
    Suitable only for n ≤ ~18, but great for benchmarking.
    """

    # ---------- TSPAlgorithm API ----------
    def solve(self, problem: TSPProblem) -> TSPResult:
        coords = problem.city_coordinates
        n = len(coords)

        dist = self._dist_matrix(coords)

        # trivial small cases
        if n <= 2:
            route = list(range(n))
            best = self._route_distance(route, dist)
            self._push_final(coords, route, best, 0.0)
            return TSPResult(best_route=route, best_distance=best)

        dp = [[math.inf] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        dp[1][0] = 0.0  # start at city 0

        start = time.time()

        # iterate over all subsets
        for mask in range(1, 1 << n):
            if cancel_event.is_set():
                break
            pause_event.wait()

            for i in range(n):
                if not (mask & (1 << i)):
                    continue  # city i not in subset
                for j in range(n):
                    if (mask & (1 << j)) or i == j:
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = dp[mask][i] + dist[i][j]
                    if new_cost < dp[new_mask][j]:
                        dp[new_mask][j] = new_cost
                        parent[new_mask][j] = i

        all_visited = (1 << n) - 1
        best_d, end_city = math.inf, -1
        for i in range(n):
            cost = dp[all_visited][i] + dist[i][0]
            if cost < best_d:
                best_d, end_city = cost, i

        # reconstruct route
        route: List[int] = []
        mask = all_visited
        curr = end_city
        while curr != -1:
            route.append(curr)
            prev = parent[mask][curr]
            mask &= ~(1 << curr)
            curr = prev
        route.reverse()

        self._push_final(coords, route, best_d, time.time() - start)
        return TSPResult(best_route=route, best_distance=best_d)

    # ---------- helpers ----------
    @staticmethod
    def _dist_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
        n = len(coords)
        m = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                m[i][j] = m[j][i] = d
        return m

    @staticmethod
    def _route_distance(route: List[int], m) -> float:
        return sum(m[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

    def _push_final(self, coords, route, dist, runtime):
        """Send a single UI update when the exact solver finishes."""
        update_queue.put(
            dict(
                algo_key="exact",
                coords=coords,
                route=route,
                distance=dist,
                runtime=runtime,
                iteration="final",
            )
        )
