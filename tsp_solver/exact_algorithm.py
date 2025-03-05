import math
from typing import List, Tuple

from tsp_solver.base_algorithm import TSPAlgorithm
from tsp_solver.models import TSPProblem, TSPResult


class TSPExactAlgorithm(TSPAlgorithm):
    """
    A Held-Karp dynamic programming approach to find an exact TSP solution.
    WARNING: This is exponential (O(n^2 * 2^n)) and only practical for small n.
    """

    def solve(self, problem: TSPProblem) -> TSPResult:
        coords = problem.city_coordinates
        n = len(coords)

        # Build distance matrix
        dist_matrix = self._compute_distance_matrix(coords)

        # Edge case: if only 1 or 2 cities, handle trivially
        if n <= 2:
            route = list(range(n))
            best_distance = self._route_distance(route, dist_matrix)
            return TSPResult(best_route=route, best_distance=best_distance)

        # dp[mask][i] will hold the minimum cost to visit the set of cities in 'mask'
        # ending at city i.
        dp = [[math.inf] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]

        # Start at city 0
        dp[1][0] = 0

        for mask in range(1 << n):
            for i in range(n):
                # If city i not in mask, skip
                if not (mask & (1 << i)):
                    continue
                # If i is in mask, try to go from i -> j
                for j in range(n):
                    if (mask & (1 << j)) or j == i:
                        continue
                    new_mask = mask | (1 << j)
                    cost = dp[mask][i] + dist_matrix[i][j]
                    if cost < dp[new_mask][j]:
                        dp[new_mask][j] = cost
                        parent[new_mask][j] = i

        # Close the loop by returning to city 0
        all_visited = (1 << n) - 1
        best_distance = math.inf
        best_end_city = -1

        for i in range(n):
            cost = dp[all_visited][i] + dist_matrix[i][0]
            if cost < best_distance:
                best_distance = cost
                best_end_city = i

        # Reconstruct route
        route = []
        mask = all_visited
        curr = best_end_city
        while curr != -1:
            route.append(curr)
            temp = parent[mask][curr]
            mask = mask & ~(1 << curr)
            curr = temp

        route.reverse()  # because we built it backwards
        return TSPResult(best_route=route, best_distance=best_distance)

    def _compute_distance_matrix(
        self, coords: List[Tuple[float, float]]
    ) -> List[List[float]]:
        n = len(coords)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    matrix[i][j] = math.sqrt(dx * dx + dy * dy)
        return matrix

    def _route_distance(
        self, route: List[int], dist_matrix: List[List[float]]
    ) -> float:
        dist = 0.0
        for i in range(len(route) - 1):
            dist += dist_matrix[route[i]][route[i + 1]]
        if len(route) > 1:
            dist += dist_matrix[route[-1]][route[0]]
        return dist
