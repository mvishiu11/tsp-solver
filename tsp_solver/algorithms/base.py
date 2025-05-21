# algorithms/base.py
"""
Common abstract base class for all TSP solvers.

Keeping this minimal lets every algorithm remain unit-testable without
bringing in any UI-specific dependencies.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from tsp_solver.models import TSPProblem, TSPResult


class TSPAlgorithm(ABC):
    """Interface every TSP solver must implement."""

    @abstractmethod
    def solve(self, problem: TSPProblem) -> TSPResult:  # pragma: no cover
        """
        Run the algorithm on *problem* and return a TSPResult containing
        the best route and its distance.  Implementations may push
        intermediate progress to the UI via utils.events.update_queue.
        """
        raise NotImplementedError
