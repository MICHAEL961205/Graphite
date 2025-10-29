# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class CheapestInsertionTwoOptSolver(BaseSolver):
    """
    Constructs an initial tour using Cheapest Insertion, then improves it with 2-opt.
    Designed to be fast and typically better than Nearest Neighbour on metric TSPs.
    """
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, time_limit: int = 100):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        # Handle trivial sizes
        if n <= 2:
            route = list(range(n)) + [0]
            return route

        # Build initial tour with Cheapest Insertion
        tour = self._cheapest_insertion(distance_matrix, start_time)
        if tour is None:
            return None

        if (time.time() - start_time) >= self.time_limit:
            return tour + [tour[0]]

        # Improve with 2-opt under remaining time
        tour = self._two_opt_improve(tour, distance_matrix, start_time)
        return tour + [tour[0]]

    def _cheapest_insertion(self, dist: np.ndarray, start_time: float) -> List[int]:
        n = len(dist)
        # Start with the farthest pair to seed a diverse initial tour
        i0, j0 = 0, 1
        farthest = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] > farthest:
                    farthest = dist[i][j]
                    i0, j0 = i, j
        tour = [i0, j0]
        remaining = set(range(n)) - set(tour)

        # Insert nodes at position minimizing marginal cost
        while remaining:
            if (time.time() - start_time) >= self.time_limit:
                break
            best_gain = float('inf')
            best_node = None
            best_pos = None

            # Evaluate insertions: between tour[k] -> tour[(k+1) % len(tour)]
            m = len(tour)
            for node in list(remaining):
                # Early exit if time exceeded inside loop
                if (time.time() - start_time) >= self.time_limit:
                    break
                for k in range(m):
                    a = tour[k]
                    b = tour[(k + 1) % m]
                    gain = dist[a][node] + dist[node][b] - dist[a][b]
                    if gain < best_gain:
                        best_gain = gain
                        best_node = node
                        best_pos = (k + 1) % m
            if best_node is None:
                break
            tour.insert(best_pos, best_node)
            remaining.remove(best_node)
        return tour

    def _two_opt_improve(self, tour: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        n = len(tour)
        improved = True
        while improved and (time.time() - start_time) < self.time_limit:
            improved = False
            for i in range(1, n - 2):
                if (time.time() - start_time) >= self.time_limit:
                    break
                a = tour[i - 1]
                b = tour[i]
                for j in range(i + 1, n):
                    if (time.time() - start_time) >= self.time_limit:
                        break
                    c = tour[j - 1]
                    d = tour[j % n]
                    old_cost = dist[a][b] + dist[c][d]
                    new_cost = dist[a][c] + dist[b][d]
                    if new_cost + 1e-12 < old_cost:
                        tour[i:j] = reversed(tour[i:j])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "CheapestInsertionTwoOptSolver",
            "description": "Cheapest Insertion + 2-opt local improvement",
            "time_limit": self.time_limit,
        }
