from typing import List, Tuple

from pydantic import BaseModel


class TSPProblem(BaseModel):
    """Represents the input data for a TSP instance."""

    city_coordinates: List[Tuple[float, float]]
    # You could add more fields (e.g., distance matrix if precomputed).


class TSPResult(BaseModel):
    """Represents the result of solving a TSP instance."""

    best_route: List[int]
    best_distance: float
    # Optionally add stats like generation_count, time_elapsed, etc.
