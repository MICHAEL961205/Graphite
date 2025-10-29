# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union, Tuple
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class LinKernighanSolver(BaseSolver):
    """
    Lin-Kernighan algorithm for TSP - focuses on quality within time limit.
    
    This is a sophisticated local search algorithm that uses variable-depth
    k-opt moves to find high-quality solutions.
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

        if n <= 2:
            return list(range(n)) + [0]

        # Start with nearest neighbor
        nn_solver = NearestNeighbourSolver()
        tour = await nn_solver.solve(formatted_problem, future_id)
        if tour is None or len(tour) < 2:
            return list(range(n)) + [0]
        
        if tour[-1] == tour[0]:
            tour = tour[:-1]

        # Apply Lin-Kernighan improvement
        tour = self._lin_kernighan(tour, distance_matrix, start_time)
        return tour + [tour[0]]

    def _lin_kernighan(self, tour: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        n = len(tour)
        best_tour = tour[:]
        best_cost = self._tour_cost(tour, dist)
        
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and (time.time() - start_time) < self.time_limit and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try Lin-Kernighan from each starting position
            for start_pos in range(n):
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                new_tour = self._lin_kernighan_from_position(tour, dist, start_pos, start_time)
                if new_tour is not None:
                    new_cost = self._tour_cost(new_tour, dist)
                    if new_cost < best_cost - 1e-12:
                        tour = new_tour
                        best_tour = new_tour[:]
                        best_cost = new_cost
                        improved = True
                        break
                        
        return best_tour

    def _lin_kernighan_from_position(self, tour: List[int], dist: np.ndarray, start_pos: int, start_time: float) -> List[int]:
        n = len(tour)
        if n < 4:
            return None
            
        # Create position mapping
        pos = {tour[i]: i for i in range(n)}
        
        # Try different k-opt moves starting from position start_pos
        for k in range(2, min(6, n//2)):  # Try 2-opt to 5-opt
            if (time.time() - start_time) >= self.time_limit:
                break
                
            new_tour = self._try_k_opt(tour, dist, start_pos, k, pos)
            if new_tour is not None:
                return new_tour
                
        return None

    def _try_k_opt(self, tour: List[int], dist: np.ndarray, start_pos: int, k: int, pos: dict) -> List[int]:
        n = len(tour)
        if n < 2 * k:
            return None
            
        # Generate k-opt moves
        for edges_to_remove in self._generate_k_opt_edges(start_pos, k, n):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            # Check if this is a valid k-opt move
            if self._is_valid_k_opt(tour, edges_to_remove, pos):
                new_tour = self._apply_k_opt(tour, edges_to_remove, dist, pos)
                if new_tour is not None:
                    return new_tour
                    
        return None

    def _generate_k_opt_edges(self, start_pos: int, k: int, n: int):
        """Generate possible k-opt edge removals"""
        edges = []
        
        # Start with edge from start_pos
        current = start_pos
        for i in range(k):
            next_pos = (current + 1) % n
            edges.append((current, next_pos))
            current = (current + 2) % n
            
        yield edges

    def _is_valid_k_opt(self, tour: List[int], edges_to_remove: List[Tuple[int, int]], pos: dict) -> bool:
        """Check if k-opt move is valid"""
        # Check that edges are alternating and form a valid path
        if len(edges_to_remove) < 2:
            return False
            
        # Simple validation - edges should be consecutive
        for i in range(len(edges_to_remove) - 1):
            if edges_to_remove[i][1] != edges_to_remove[i + 1][0]:
                return False
                
        return True

    def _apply_k_opt(self, tour: List[int], edges_to_remove: List[Tuple[int, int]], dist: np.ndarray, pos: dict) -> List[int]:
        """Apply k-opt move if it improves the tour"""
        if len(edges_to_remove) < 2:
            return None
            
        # Calculate cost change
        old_cost = sum(dist[tour[e[0]]][tour[e[1]]] for e in edges_to_remove)
        
        # Generate new edges
        new_edges = []
        for i in range(len(edges_to_remove)):
            if i % 2 == 0:  # Even indices: reverse the edge
                new_edges.append((edges_to_remove[i][1], edges_to_remove[i][0]))
            else:  # Odd indices: keep the edge
                new_edges.append(edges_to_remove[i])
                
        new_cost = sum(dist[tour[e[0]]][tour[e[1]]] for e in new_edges)
        
        if new_cost >= old_cost - 1e-12:
            return None
            
        # Apply the move
        new_tour = tour[:]
        for i, (start, end) in enumerate(edges_to_remove):
            if i % 2 == 0:  # Reverse this segment
                segment_start = start
                segment_end = end
                if segment_start < segment_end:
                    new_tour[segment_start:segment_end+1] = reversed(new_tour[segment_start:segment_end+1])
                else:
                    # Handle wraparound case
                    segment = new_tour[segment_start:] + new_tour[:segment_end+1]
                    segment = list(reversed(segment))
                    new_tour[segment_start:] = segment[:len(new_tour)-segment_start]
                    new_tour[:segment_end+1] = segment[len(new_tour)-segment_start:]
                    
        return new_tour

    def _tour_cost(self, tour: List[int], dist: np.ndarray) -> float:
        """Calculate total tour cost"""
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
            "name": "LinKernighanSolver",
            "description": "Lin-Kernighan algorithm (quality-focused)",
            "time_limit": self.time_limit
        }
