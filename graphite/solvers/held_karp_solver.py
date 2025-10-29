# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union, Dict, Tuple
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class HeldKarpSolver(BaseSolver):
    """
    Held-Karp algorithm for TSP - exact solution.
    
    Uses dynamic programming with memoization to find
    optimal solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_size: int = 20):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_size = max_size

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()
        hard_limit = self.time_limit  # Hard limit at exactly time_limit

        if n <= 2:
            return list(range(n)) + [0]

        # For small problems, use exact Held-Karp
        if n <= self.max_size:
            return self._exact_held_karp(distance_matrix, start_time, hard_limit)
        else:
            # For larger problems, use heuristic Held-Karp
            return self._heuristic_held_karp(distance_matrix, start_time, hard_limit)

    def _exact_held_karp(self, dist: np.ndarray, start_time: float, hard_limit: float) -> List[int]:
        """Exact Held-Karp for small problems"""
        n = len(dist)
        
        # Memoization table: (subset, last_city) -> (cost, parent)
        memo = {}
        
        # Base case: single city
        for i in range(n):
            memo[(frozenset([i]), i)] = (0, None)
        
        # Fill memoization table
        for subset_size in range(2, n + 1):
            if (time.time() - start_time) >= hard_limit:
                break
                
            for subset in self._generate_subsets(range(n), subset_size):
                if 0 not in subset:  # Must include city 0
                    continue
                    
                for k in subset:
                    if k == 0:
                        continue
                    
                    subset_without_k = subset - {k}
                    min_cost = float('inf')
                    min_parent = None
                    
                    for m in subset_without_k:
                        if m == 0 and len(subset_without_k) > 1:
                            continue
                        
                        cost, _ = memo.get((subset_without_k, m), (float('inf'), None))
                        if cost + dist[m][k] < min_cost:
                            min_cost = cost + dist[m][k]
                            min_parent = m
                    
                    memo[(subset, k)] = (min_cost, min_parent)
        
        # Find optimal tour
        all_cities = frozenset(range(n))
        min_cost = float('inf')
        last_city = None
        
        for k in range(1, n):
            cost, _ = memo.get((all_cities, k), (float('inf'), None))
            if cost + dist[k][0] < min_cost:
                min_cost = cost + dist[k][0]
                last_city = k
        
        if last_city is None:
            return list(range(n)) + [0]
        
        # Reconstruct tour
        tour = self._reconstruct_tour(memo, all_cities, last_city)
        return tour + [tour[0]]

    def _heuristic_held_karp(self, dist: np.ndarray, start_time: float, hard_limit: float) -> List[int]:
        """Heuristic Held-Karp for larger problems"""
        n = len(dist)
        
        # Start with nearest neighbor
        solution = self._nearest_neighbor(dist, start_time)
        if solution is None:
            return list(range(n)) + [0]
        
        # Apply 2-opt improvements
        solution = self._two_opt_improve(solution, dist, start_time, hard_limit)
        
        # Use Held-Karp on smaller subproblems
        best_solution = solution
        best_cost = self._calculate_cost(best_solution, dist)
        
        # Try different starting points
        for start_city in range(min(5, n)):
            if (time.time() - start_time) >= hard_limit:
                break
                
            # Create subproblem with subset of cities
            subset_size = min(12, n)
            cities = list(range(n))
            random.shuffle(cities)
            subset = cities[:subset_size]
            
            if 0 not in subset:
                subset[0] = 0  # Ensure city 0 is included
            
            # Solve subproblem
            subproblem_solution = self._solve_subproblem(dist, subset, start_time)
            if subproblem_solution is not None:
                # Extend to full solution
                full_solution = self._extend_solution(subproblem_solution, dist, start_time)
                if full_solution is not None:
                    cost = self._calculate_cost(full_solution, dist)
                    if cost < best_cost - 1e-12:
                        best_solution = full_solution
                        best_cost = cost
        
        return best_solution + [best_solution[0]]

    def _solve_subproblem(self, dist: np.ndarray, subset: List[int], start_time: float, hard_limit: float) -> List[int]:
        """Solve Held-Karp for a subset of cities"""
        n = len(subset)
        if n <= 2:
            return subset
        
        # Create distance matrix for subset
        sub_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sub_dist[i][j] = dist[subset[i]][subset[j]]
        
        # Solve using exact Held-Karp
        memo = {}
        
        # Base case
        for i in range(n):
            memo[(frozenset([i]), i)] = (0, None)
        
        # Fill memoization table
        for subset_size in range(2, n + 1):
            if (time.time() - start_time) >= hard_limit:
                break
                
            for sub_subset in self._generate_subsets(range(n), subset_size):
                if 0 not in sub_subset:
                    continue
                    
                for k in sub_subset:
                    if k == 0:
                        continue
                    
                    sub_subset_without_k = sub_subset - {k}
                    min_cost = float('inf')
                    min_parent = None
                    
                    for m in sub_subset_without_k:
                        if m == 0 and len(sub_subset_without_k) > 1:
                            continue
                        
                        cost, _ = memo.get((sub_subset_without_k, m), (float('inf'), None))
                        if cost + sub_dist[m][k] < min_cost:
                            min_cost = cost + sub_dist[m][k]
                            min_parent = m
                    
                    memo[(sub_subset, k)] = (min_cost, min_parent)
        
        # Find optimal tour
        all_cities = frozenset(range(n))
        min_cost = float('inf')
        last_city = None
        
        for k in range(1, n):
            cost, _ = memo.get((all_cities, k), (float('inf'), None))
            if cost + sub_dist[k][0] < min_cost:
                min_cost = cost + sub_dist[k][0]
                last_city = k
        
        if last_city is None:
            return subset
        
        # Reconstruct tour
        tour_indices = self._reconstruct_tour(memo, all_cities, last_city)
        return [subset[i] for i in tour_indices]

    def _extend_solution(self, partial_solution: List[int], dist: np.ndarray, start_time: float, hard_limit: float) -> List[int]:
        """Extend partial solution to full solution"""
        n = len(dist)
        remaining = set(range(n)) - set(partial_solution)
        
        if not remaining:
            return partial_solution
        
        # Use nearest neighbor to extend
        solution = partial_solution[:]
        current = solution[-1]
        
        while remaining and (time.time() - start_time) < self.time_limit:
            nearest = None
            best_dist = float('inf')
            for city in remaining:
                if dist[current][city] < best_dist:
                    best_dist = dist[current][city]
                    nearest = city
            if nearest is None:
                break
            solution.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return solution

    def _generate_subsets(self, cities: List[int], size: int) -> List[frozenset]:
        """Generate all subsets of given size"""
        if size == 0:
            return [frozenset()]
        if size > len(cities):
            return []
        
        subsets = []
        for i in range(len(cities)):
            if size == 1:
                subsets.append(frozenset([cities[i]]))
            else:
                for subset in self._generate_subsets(cities[i+1:], size-1):
                    subsets.append(frozenset([cities[i]]) | subset)
        
        return subsets

    def _reconstruct_tour(self, memo: Dict, subset: frozenset, last_city: int) -> List[int]:
        """Reconstruct tour from memoization table"""
        tour = []
        current_subset = subset
        current_city = last_city
        
        while current_city is not None:
            tour.append(current_city)
            _, parent = memo.get((current_subset, current_city), (0, None))
            if parent is not None:
                current_subset = current_subset - {current_city}
                current_city = parent
            else:
                break
        
        return tour[::-1]  # Reverse to get correct order

    def _nearest_neighbor(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Nearest neighbor construction"""
        n = len(dist)
        tour = [0]
        visited = {0}
        current = 0
        
        for _ in range(n - 1):
            if (time.time() - start_time) >= hard_limit:
                break
            nearest = None
            best_dist = float('inf')
            for j in range(n):
                if j not in visited and dist[current][j] < best_dist:
                    best_dist = dist[current][j]
                    nearest = j
            if nearest is None:
                break
            tour.append(nearest)
            visited.add(nearest)
            current = nearest
        return tour

    def _two_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float, hard_limit: float) -> List[int]:
        """2-opt local improvement"""
        n = len(solution)
        improved_solution = solution[:]
        improved = True
        iterations = 0
        max_iterations = 20
        
        while improved and iterations < max_iterations and (time.time() - start_time) < self.time_limit:
            improved = False
            iterations += 1
            
            for i in range(1, n - 2):
                if (time.time() - start_time) >= hard_limit:
                    break
                for j in range(i + 1, n):
                    if (time.time() - start_time) >= hard_limit:
                        break
                    a, b = improved_solution[i-1], improved_solution[i]
                    c, d = improved_solution[j-1], improved_solution[j]
                    if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-12:
                        improved_solution[i:j] = reversed(improved_solution[i:j])
                        improved = True
                        break
                if improved:
                    break
        
        return improved_solution

    def _calculate_cost(self, solution: List[int], dist: np.ndarray) -> float:
        """Calculate tour cost"""
        if len(solution) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(solution) - 1):
            cost += dist[solution[i]][solution[i + 1]]
        return cost

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "HeldKarpSolver",
            "description": "Held-Karp dynamic programming algorithm",
            "time_limit": self.time_limit,
            "max_size": self.max_size
        }
