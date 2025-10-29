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

class GRASPSolver(BaseSolver):
    """
    GRASP (Greedy Randomized Adaptive Search Procedure) for TSP - quality-focused.
    
    Uses randomized greedy construction followed by local search
    to find high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_iterations: int = 1000,
                 alpha: float = 0.3, beta: float = 0.1):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.alpha = alpha  # Greediness parameter
        self.beta = beta    # Randomness parameter

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        best_solution = None
        best_cost = float('inf')
        
        # GRASP main loop
        iteration = 0
        while iteration < self.max_iterations and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
            
            # Construction phase
            solution = self._greedy_randomized_construction(distance_matrix, start_time)
            if solution is None or len(solution) != n:
                iteration += 1
                continue
            
            # Local search phase
            improved_solution = self._local_search(solution, distance_matrix, start_time)
            improved_cost = self._calculate_cost(improved_solution, distance_matrix)
            
            # Update best solution
            if improved_cost < best_cost - 1e-12:
                best_solution = improved_solution[:]
                best_cost = improved_cost
            
            iteration += 1
        
        if best_solution is None:
            # Fallback to nearest neighbor
            nn_solver = NearestNeighbourSolver()
            fallback_solution = await nn_solver.solve(formatted_problem, future_id)
            if fallback_solution and len(fallback_solution) > 1:
                if fallback_solution[-1] == fallback_solution[0]:
                    fallback_solution = fallback_solution[:-1]
                return fallback_solution + [fallback_solution[0]]
            return list(range(n)) + [0]
        
        return best_solution + [best_solution[0]]

    def _greedy_randomized_construction(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Greedy randomized construction"""
        n = len(dist)
        solution = []
        remaining = set(range(n))
        
        # Start with random city
        current = random.choice(list(remaining))
        solution.append(current)
        remaining.remove(current)
        
        while remaining and (time.time() - start_time) < self.time_limit:
            # Calculate costs for all remaining cities
            costs = []
            cities = []
            for city in remaining:
                cost = dist[current][city]
                costs.append(cost)
                cities.append(city)
            
            if not costs:
                break
            
            # Create restricted candidate list (RCL)
            min_cost = min(costs)
            max_cost = max(costs)
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            
            rcl = [cities[i] for i in range(len(cities)) if costs[i] <= threshold]
            
            if not rcl:
                rcl = cities
            
            # Randomly select from RCL
            if random.random() < self.beta:
                # Pure random selection
                next_city = random.choice(rcl)
            else:
                # Greedy selection from RCL
                best_cost = float('inf')
                next_city = None
                for city in rcl:
                    cost = dist[current][city]
                    if cost < best_cost:
                        best_cost = cost
                        next_city = city
                
                if next_city is None:
                    next_city = random.choice(rcl)
            
            solution.append(next_city)
            remaining.remove(next_city)
            current = next_city
        
        return solution

    def _local_search(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Multi-strategy local search"""
        n = len(solution)
        current_solution = solution[:]
        improved = True
        iterations = 0
        max_iterations = 15
        
        while improved and iterations < max_iterations and (time.time() - start_time) < self.time_limit:
            improved = False
            iterations += 1
            
            # Try different local search strategies
            strategies = ['2opt', '3opt', 'or_opt', 'swap']
            
            for strategy in strategies:
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                if strategy == '2opt':
                    if self._two_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == '3opt':
                    if self._three_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == 'or_opt':
                    if self._or_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == 'swap':
                    if self._swap_improve(current_solution, dist, start_time):
                        improved = True
                        break
        
        return current_solution

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
                        if self._calculate_cost(config, dist) < self._calculate_cost(tour, dist) - 1e-12:
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
                    if self._calculate_cost(new_tour, dist) < self._calculate_cost(tour, dist) - 1e-12:
                        tour[:] = new_tour
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    def _swap_improve(self, tour: List[int], dist: np.ndarray, start_time: float) -> bool:
        """Swap local improvement"""
        n = len(tour)
        improved = False
        
        for i in range(n - 1):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= self.time_limit:
                    break
                # Calculate cost change
                a, b = tour[i-1], tour[i]
                c, d = tour[(i+1) % n], tour[j-1]
                e, f = tour[j], tour[(j+1) % n]
                
                old_cost = dist[a][b] + dist[b][c] + dist[d][e] + dist[e][f]
                new_cost = dist[a][e] + dist[e][c] + dist[d][b] + dist[b][f]
                
                if new_cost < old_cost - 1e-12:
                    tour[i], tour[j] = tour[j], tour[i]
                    improved = True
                    break
            if improved:
                break
        return improved

    def _calculate_cost(self, tour: List[int], dist: np.ndarray) -> float:
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
            "name": "GRASPSolver",
            "description": "GRASP (Greedy Randomized Adaptive Search)",
            "time_limit": self.time_limit,
            "max_iterations": self.max_iterations,
            "alpha": self.alpha,
            "beta": self.beta
        }
