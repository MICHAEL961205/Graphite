# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union, Tuple, Optional
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import heapq
import random

class BranchAndBoundSolver(BaseSolver):
    """
    Branch and Bound algorithm for TSP - exact solution.
    
    Uses branch and bound with lower bound heuristics
    to find optimal solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_nodes: int = 10000):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_nodes = max_nodes

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # For small problems, use exact branch and bound
        if n <= 12:
            return self._exact_branch_and_bound(distance_matrix, start_time)
        else:
            # For larger problems, use heuristic branch and bound
            return self._heuristic_branch_and_bound(distance_matrix, start_time)

    def _exact_branch_and_bound(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Exact branch and bound for small problems"""
        n = len(dist)
        
        # Initialize with best known solution
        best_solution = self._nearest_neighbor(dist, start_time)
        if best_solution is None:
            return list(range(n)) + [0]
        
        best_cost = self._calculate_cost(best_solution, dist)
        
        # Priority queue for nodes (lower bound, node)
        pq = []
        
        # Root node: empty partial tour
        root_node = {
            'partial_tour': [],
            'remaining': set(range(n)),
            'lower_bound': self._calculate_lower_bound([], set(range(n)), dist),
            'cost': 0
        }
        heapq.heappush(pq, (root_node['lower_bound'], root_node))
        
        nodes_explored = 0
        
        while pq and nodes_explored < self.max_nodes and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
                
            lower_bound, node = heapq.heappop(pq)
            nodes_explored += 1
            
            # Prune if lower bound is worse than best solution
            if lower_bound >= best_cost:
                continue
            
            # If partial tour is complete, check if it's better
            if len(node['partial_tour']) == n:
                if node['cost'] < best_cost - 1e-12:
                    best_solution = node['partial_tour'][:]
                    best_cost = node['cost']
                continue
            
            # Branch: add each remaining city
            for city in node['remaining']:
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                new_partial_tour = node['partial_tour'] + [city]
                new_remaining = node['remaining'] - {city}
                
                # Calculate cost of new partial tour
                new_cost = node['cost']
                if len(new_partial_tour) > 1:
                    new_cost += dist[new_partial_tour[-2]][new_partial_tour[-1]]
                
                # Calculate lower bound
                new_lower_bound = self._calculate_lower_bound(new_partial_tour, new_remaining, dist)
                
                # Only add if lower bound is promising
                if new_lower_bound < best_cost:
                    new_node = {
                        'partial_tour': new_partial_tour,
                        'remaining': new_remaining,
                        'lower_bound': new_lower_bound,
                        'cost': new_cost
                    }
                    heapq.heappush(pq, (new_lower_bound, new_node))
        
        return best_solution + [best_solution[0]]

    def _heuristic_branch_and_bound(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Heuristic branch and bound for larger problems"""
        n = len(dist)
        
        # Start with nearest neighbor
        solution = self._nearest_neighbor(dist, start_time)
        if solution is None:
            return list(range(n)) + [0]
        
        best_solution = solution
        best_cost = self._calculate_cost(best_solution, dist)
        
        # Apply 2-opt improvements
        best_solution = self._two_opt_improve(best_solution, dist, start_time)
        best_cost = self._calculate_cost(best_solution, dist)
        
        # Limited branch and bound search
        pq = []
        root_node = {
            'partial_tour': [],
            'remaining': set(range(n)),
            'lower_bound': self._calculate_lower_bound([], set(range(n)), dist),
            'cost': 0
        }
        heapq.heappush(pq, (root_node['lower_bound'], root_node))
        
        nodes_explored = 0
        max_depth = min(8, n)  # Limit depth for larger problems
        
        while pq and nodes_explored < self.max_nodes and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
                
            lower_bound, node = heapq.heappop(pq)
            nodes_explored += 1
            
            # Prune if lower bound is worse than best solution
            if lower_bound >= best_cost:
                continue
            
            # If partial tour is complete, check if it's better
            if len(node['partial_tour']) == n:
                if node['cost'] < best_cost - 1e-12:
                    best_solution = node['partial_tour'][:]
                    best_cost = node['cost']
                continue
            
            # Limit depth for larger problems
            if len(node['partial_tour']) >= max_depth:
                continue
            
            # Branch: add each remaining city
            for city in node['remaining']:
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                new_partial_tour = node['partial_tour'] + [city]
                new_remaining = node['remaining'] - {city}
                
                # Calculate cost of new partial tour
                new_cost = node['cost']
                if len(new_partial_tour) > 1:
                    new_cost += dist[new_partial_tour[-2]][new_partial_tour[-1]]
                
                # Calculate lower bound
                new_lower_bound = self._calculate_lower_bound(new_partial_tour, new_remaining, dist)
                
                # Only add if lower bound is promising
                if new_lower_bound < best_cost:
                    new_node = {
                        'partial_tour': new_partial_tour,
                        'remaining': new_remaining,
                        'lower_bound': new_lower_bound,
                        'cost': new_cost
                    }
                    heapq.heappush(pq, (new_lower_bound, new_node))
        
        return best_solution + [best_solution[0]]

    def _calculate_lower_bound(self, partial_tour: List[int], remaining: set, dist: np.ndarray) -> float:
        """Calculate lower bound for partial tour"""
        if not partial_tour:
            # Use minimum spanning tree as lower bound
            return self._mst_lower_bound(remaining, dist)
        
        n = len(dist)
        current_cost = 0
        
        # Cost of partial tour
        for i in range(len(partial_tour) - 1):
            current_cost += dist[partial_tour[i]][partial_tour[i + 1]]
        
        if not remaining:
            return current_cost
        
        # Add minimum cost to complete the tour
        # From last city in partial tour to first remaining city
        if remaining:
            min_outgoing = min(dist[partial_tour[-1]][city] for city in remaining)
            current_cost += min_outgoing
        
        # Add minimum spanning tree of remaining cities
        mst_cost = self._mst_lower_bound(remaining, dist)
        current_cost += mst_cost
        
        # Add minimum cost from remaining cities back to start
        if remaining:
            min_incoming = min(dist[city][partial_tour[0]] for city in remaining)
            current_cost += min_incoming
        
        return current_cost

    def _mst_lower_bound(self, cities: set, dist: np.ndarray) -> float:
        """Calculate minimum spanning tree lower bound"""
        if len(cities) <= 1:
            return 0.0
        
        cities_list = list(cities)
        n = len(cities_list)
        
        # Prim's algorithm for MST
        mst_cost = 0
        visited = {cities_list[0]}
        remaining = set(cities_list[1:])
        
        while remaining:
            min_edge = float('inf')
            min_city = None
            
            for visited_city in visited:
                for remaining_city in remaining:
                    edge_cost = dist[visited_city][remaining_city]
                    if edge_cost < min_edge:
                        min_edge = edge_cost
                        min_city = remaining_city
            
            if min_city is not None:
                mst_cost += min_edge
                visited.add(min_city)
                remaining.remove(min_city)
            else:
                break
        
        return mst_cost

    def _nearest_neighbor(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Nearest neighbor construction"""
        n = len(dist)
        tour = [0]
        visited = {0}
        current = 0
        
        for _ in range(n - 1):
            if (time.time() - start_time) >= self.time_limit:
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

    def _two_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
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
                if (time.time() - start_time) >= self.time_limit:
                    break
                for j in range(i + 1, n):
                    if (time.time() - start_time) >= self.time_limit:
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
            "name": "BranchAndBoundSolver",
            "description": "Branch and Bound exact algorithm",
            "time_limit": self.time_limit,
            "max_nodes": self.max_nodes
        }
