# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class FarthestInsertionTwoOptSolver(BaseSolver):
    """
    Farthest Insertion + 2-opt for TSP.
    
    Farthest Insertion selects the unvisited city farthest from the current tour,
    which often produces better initial tours than nearest insertion.
    Then improves with 2-opt.
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
        effective_time_limit = self.time_limit * 0.95

        if n <= 2:
            return list(range(n)) + [0]

        # Farthest Insertion (faster version)
        if n > 1000:
            nn_solver = NearestNeighbourSolver()
            tour = await nn_solver.solve(formatted_problem, future_id)
            if tour and len(tour) > 1:
                if tour[-1] == tour[0]:
                    tour = tour[:-1]
            else:
                tour = list(range(n))
        else:
            tour = self._farthest_insertion(distance_matrix, start_time, effective_time_limit * 0.7)
            if tour is None or len(tour) < n:
                nn_solver = NearestNeighbourSolver()
                tour = await nn_solver.solve(formatted_problem, future_id)
                if tour and len(tour) > 1:
                    if tour[-1] == tour[0]:
                        tour = tour[:-1]

        if (time.time() - start_time) >= effective_time_limit:
            return tour + [tour[0]] if tour else list(range(n)) + [0]

        # Quick 2-opt improvement
        if n <= 500:
            tour = self._two_opt_quick(tour, distance_matrix, start_time, effective_time_limit)
        return tour + [tour[0]] if tour else list(range(n)) + [0]

    def _farthest_insertion(self, dist: np.ndarray, start_time: float, time_limit: float) -> List[int]:
        n = len(dist)
        # Start with two farthest nodes
        best_dist = -1
        i0, j0 = 0, 1
        search_limit = min(n, 100)
        for i in range(search_limit):
            if (time.time() - start_time) >= time_limit:
                break
            for j in range(i + 1, min(i + 50, n)):
                if dist[i][j] > best_dist:
                    best_dist = dist[i][j]
                    i0, j0 = i, j
        
        tour = [i0, j0]
        remaining = set(range(n)) - set(tour)
        check_counter = 0

        while remaining:
            check_counter += 1
            if check_counter % 10 == 0 and (time.time() - start_time) >= time_limit:
                break
                
            # Find unvisited node farthest from tour
            best_dist_from_tour = -1
            best_node = None
            remaining_list = list(remaining)[:min(100, len(remaining))]
            for node in remaining_list:
                min_dist_to_tour = min([dist[node][t] for t in tour])
                if min_dist_to_tour > best_dist_from_tour:
                    best_dist_from_tour = min_dist_to_tour
                    best_node = node

            if best_node is None:
                if remaining:
                    tour.extend(list(remaining))
                    break
                break

            # Insert at position that minimizes cost
            best_pos = 0
            best_cost = float('inf')
            for k in range(len(tour)):
                a, b = tour[k], tour[(k + 1) % len(tour)]
                cost = dist[a][best_node] + dist[best_node][b] - dist[a][b]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (k + 1) % len(tour)
            
            tour.insert(best_pos, best_node)
            remaining.remove(best_node)
        
        return tour

    def _two_opt_quick(self, tour: List[int], dist: np.ndarray, start_time: float, time_limit: float) -> List[int]:
        n = len(tour)
        improved = True
        iterations = 0
        max_iterations = 3
        
        while improved and iterations < max_iterations:
            improved = False
            if (time.time() - start_time) >= time_limit:
                break
                
            for i in range(1, min(n - 2, 50)):
                if (time.time() - start_time) >= time_limit:
                    break
                a = tour[i - 1]
                b = tour[i]
                for j in range(i + 1, min(i + 10, n)):
                    if (time.time() - start_time) >= time_limit:
                        break
                    c = tour[j - 1]
                    d = tour[j % n]
                    if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-12:
                        tour[i:j] = reversed(tour[i:j])
                        improved = True
                        break
                if improved:
                    break
            iterations += 1
        return tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "FarthestInsertionTwoOptSolver",
            "description": "Farthest Insertion + 2-opt",
            "time_limit": self.time_limit
        }

