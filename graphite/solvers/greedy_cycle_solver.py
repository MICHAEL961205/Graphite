# The MIT License (MIT)

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time

class GreedyCycleSolver(BaseSolver):
    """
    Greedy cycle heuristic: start with a small cycle and repeatedly insert the
    city whose insertion increases the tour length the least (greedy insertion).
    Distinct from cheapest insertion by seeding a small initial cycle.
    """

    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, time_limit: int = 100):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        dist = formatted_problem
        n = len(dist)
        start_time = time.time()
        if n <= 2:
            return list(range(n)) + [0]

        # Seed with a triangle: choose two farthest from 0
        i0 = 0
        j = int(np.argmax(dist[i0]))
        k = int(np.argmax(dist[j]))
        tour = [i0, j, k]
        remaining = set(range(n)) - set(tour)

        # Insert nodes greedily by minimal insertion cost
        while remaining and (time.time() - start_time) < self.time_limit:
            best_increase = float('inf')
            best_city = None
            best_pos = None
            for city in list(remaining)[:min(len(remaining), 500)]:
                # try all insertion arcs
                for pos in range(len(tour)):
                    a = tour[pos]
                    b = tour[(pos + 1) % len(tour)] if pos + 1 < len(tour) else tour[0]
                    increase = dist[a][city] + dist[city][b] - dist[a][b]
                    if increase < best_increase:
                        best_increase = increase
                        best_city = city
                        best_pos = (pos + 1) % len(tour)
            if best_city is None:
                break
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)

        tour.append(tour[0])
        return tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "GreedyCycleSolver",
            "description": "Greedy cycle insertion heuristic for TSP",
            "time_limit": self.time_limit
        }


