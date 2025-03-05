from abc import ABC, abstractmethod

from tsp_solver.models import TSPProblem, TSPResult


class TSPAlgorithm(ABC):
    """Abstract base class for TSP solvers."""

    @abstractmethod
    def solve(self, problem: TSPProblem) -> TSPResult:
        """Given a TSP problem, return the solution."""
        pass
