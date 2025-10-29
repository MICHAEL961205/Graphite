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

class VariableNeighborhoodSolver(BaseSolver):
    """
    Variable Neighborhood Search for TSP - quality-focused.
    
    Uses multiple neighborhood structures to escape local optima
    and find high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_neighborhoods: int = 5):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_neighborhoods = max_neighborhoods

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # Start with nearest neighbor
        nn_solver = NearestNeighbourSolver()
        current_tour = await nn_solver.solve(formatted_problem, future_id)
        if current_tour is None or len(current_tour) < 2:
            return list(range(n)) + [0]
        
        if current_tour[-1] == current_tour[0]:
            current_tour = current_tour[:-1]

        # Variable Neighborhood Search
        best_tour = self._variable_neighborhood_search(current_tour, distance_matrix, start_time)
        return best_tour + [best_tour[0]]

    def _variable_neighborhood_search(self, initial_tour: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        n = len(initial_tour)
        current_tour = initial_tour[:]
        best_tour = initial_tour[:]
        best_cost = self._tour_cost(current_tour, dist)
        
        k = 1  # Current neighborhood
        max_k = min(self.max_neighborhoods, n // 2)
        
        while k <= max_k and (time.time() - start_time) < self.time_limit:
            improved = True
            while improved and (time.time() - start_time) < self.time_limit:
                improved = False
                
                # Shaking: generate random solution in k-th neighborhood
                shaken_tour = self._shaking(current_tour, k, dist, start_time)
                if shaken_tour is None:
                    break
                
                # Local search in k-th neighborhood
                local_tour = self._local_search(shaken_tour, k, dist, start_time)
                if local_tour is None:
                    break
                
                local_cost = self._tour_cost(local_tour, dist)
                current_cost = self._tour_cost(current_tour, dist)
                
                if local_cost < current_cost - 1e-12:
                    current_tour = local_tour
                    improved = True
                    
                    if local_cost < best_cost - 1e-12:
                        best_tour = local_tour[:]
                        best_cost = local_cost
                        k = 1  # Reset to first neighborhood
                    else:
                        k += 1  # Move to next neighborhood
                else:
                    k += 1  # Move to next neighborhood
                    
        return best_tour

    def _shaking(self, tour: List[int], k: int, dist: np.ndarray, start_time: float) -> List[int]:
        """Generate random solution in k-th neighborhood"""
        n = len(tour)
        if n < 2 * k:
            return tour
            
        shaken_tour = tour[:]
        
        # Apply k random moves
        for _ in range(k):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            move_type = random.choice(['swap', 'insertion', 'inversion'])
            
            if move_type == 'swap':
                i, j = random.sample(range(n), 2)
                shaken_tour[i], shaken_tour[j] = shaken_tour[j], shaken_tour[i]
            elif move_type == 'insertion':
                i, j = random.sample(range(n), 2)
                if i < j:
                    shaken_tour.insert(j, shaken_tour.pop(i))
                else:
                    shaken_tour.insert(i, shaken_tour.pop(j))
            elif move_type == 'inversion':
                i, j = sorted(random.sample(range(n), 2))
                shaken_tour[i:j+1] = reversed(shaken_tour[i:j+1])
                
        return shaken_tour

    def _local_search(self, tour: List[int], k: int, dist: np.ndarray, start_time: float) -> List[int]:
        """Local search in k-th neighborhood"""
        n = len(tour)
        current_tour = tour[:]
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations and (time.time() - start_time) < self.time_limit:
            improved = False
            iterations += 1
            
            # Try different moves based on neighborhood size
            if k == 1:
                # 2-opt moves
                improved = self._two_opt_improve(current_tour, dist, start_time)
            elif k == 2:
                # 3-opt moves
                improved = self._three_opt_improve(current_tour, dist, start_time)
            elif k == 3:
                # Or-opt moves
                improved = self._or_opt_improve(current_tour, dist, start_time)
            else:
                # General k-opt moves
                improved = self._general_k_opt_improve(current_tour, k, dist, start_time)
                
        return current_tour

    def _two_opt_improve(self, tour: List[int], dist: np.ndarray, start_time: float) -> bool:
        """2-opt local improvement"""
        n = len(tour)
        improved = False
        
        for i in range(1, n - 2):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= self.time_limit:
                    break
                a, b = tour[i-1], tour[i]
                c, d = tour[j-1], tour[j]
                if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-12:
                    tour[i:j] = reversed(tour[i:j])
                    improved = True
                    break
            if improved:
                break
        return improved

    def _three_opt_improve(self, tour: List[int], dist: np.ndarray, start_time: float) -> bool:
        """3-opt local improvement"""
        n = len(tour)
        if n < 6:
            return self._two_opt_improve(tour, dist, start_time)
            
        improved = False
        for i in range(1, n - 4):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, n - 2):
                if (time.time() - start_time) >= self.time_limit:
                    break
                for k in range(j + 1, n):
                    if (time.time() - start_time) >= self.time_limit:
                        break
                    # Try different 3-opt configurations
                    configs = [
                        tour[:i] + tour[j:k] + tour[i:j] + tour[k:],
                        tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:],
                        tour[:i] + tour[j:k] + tour[i:j][::-1] + tour[k:],
                        tour[:i] + tour[j:k][::-1] + tour[i:j][::-1] + tour[k:]
                    ]
                    
                    for config in configs:
                        if self._tour_cost(config, dist) < self._tour_cost(tour, dist) - 1e-12:
                            tour[:] = config
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return improved

    def _or_opt_improve(self, tour: List[int], dist: np.ndarray, start_time: float) -> bool:
        """Or-opt local improvement"""
        n = len(tour)
        improved = False
        
        for segment_length in range(1, min(4, n // 2)):
            if (time.time() - start_time) >= self.time_limit:
                break
            for start in range(n - segment_length + 1):
                if (time.time() - start_time) >= self.time_limit:
                    break
                segment = tour[start:start + segment_length]
                remaining = tour[:start] + tour[start + segment_length:]
                
                for insert_pos in range(len(remaining) + 1):
                    if (time.time() - start_time) >= self.time_limit:
                        break
                    new_tour = remaining[:insert_pos] + segment + remaining[insert_pos:]
                    if self._tour_cost(new_tour, dist) < self._tour_cost(tour, dist) - 1e-12:
                        tour[:] = new_tour
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    def _general_k_opt_improve(self, tour: List[int], k: int, dist: np.ndarray, start_time: float) -> bool:
        """General k-opt improvement"""
        # Simplified implementation - use 2-opt for k > 3
        if k <= 3:
            return self._three_opt_improve(tour, dist, start_time)
        else:
            return self._two_opt_improve(tour, dist, start_time)

    def _tour_cost(self, tour: List[int], dist: np.ndarray) -> float:
        """Calculate tour cost"""
        if len(tour) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += dist[tour[i]][tour[i + 1]]
        return cost

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "VariableNeighborhoodSolver",
            "description": "Variable Neighborhood Search (quality-focused)",
            "time_limit": self.time_limit,
            "max_neighborhoods": self.max_neighborhoods
        }
