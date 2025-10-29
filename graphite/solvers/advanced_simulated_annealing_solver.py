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
import math

class AdvancedSimulatedAnnealingSolver(BaseSolver):
    """
    Advanced Simulated Annealing for TSP - quality-focused.
    
    Uses sophisticated cooling schedules and multiple neighborhood operators
    to find high-quality solutions within the time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, initial_temp: float = 10000.0, 
                 final_temp: float = 0.1, cooling_schedule: str = 'exponential'):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_schedule = cooling_schedule

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

        # Advanced simulated annealing
        best_tour = self._advanced_simulated_annealing(current_tour, distance_matrix, start_time)
        return best_tour + [best_tour[0]]

    def _advanced_simulated_annealing(self, initial_tour: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        n = len(initial_tour)
        current_tour = initial_tour[:]
        best_tour = initial_tour[:]
        best_cost = self._tour_cost(current_tour, dist)
        current_cost = best_cost
        
        # Adaptive temperature parameters
        temperature = self.initial_temp
        final_temp = self.final_temp
        
        # Cooling schedule parameters
        if self.cooling_schedule == 'exponential':
            alpha = 0.95
        elif self.cooling_schedule == 'linear':
            alpha = (final_temp / self.initial_temp) ** (1.0 / (self.time_limit * 10))
        else:  # geometric
            alpha = 0.99
            
        # Statistics for adaptive parameters
        accepted_moves = 0
        total_moves = 0
        iteration = 0
        
        while temperature > final_temp and (time.time() - start_time) < self.time_limit:
            iteration += 1
            
            # Generate neighbor using multiple operators
            neighbor_tour = self._generate_neighbor(current_tour, dist, iteration)
            if neighbor_tour is None:
                continue
                
            neighbor_cost = self._tour_cost(neighbor_tour, dist)
            delta = neighbor_cost - current_cost
            
            # Accept or reject move
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_tour = neighbor_tour
                current_cost = neighbor_cost
                accepted_moves += 1
                
                # Update best solution
                if current_cost < best_cost:
                    best_tour = current_tour[:]
                    best_cost = current_cost
                    
            total_moves += 1
            
            # Adaptive temperature adjustment
            if iteration % 100 == 0:
                acceptance_rate = accepted_moves / max(total_moves, 1)
                if acceptance_rate < 0.1:
                    temperature *= 1.1  # Increase temperature
                elif acceptance_rate > 0.5:
                    temperature *= 0.9  # Decrease temperature
                    
            # Cool down
            if self.cooling_schedule == 'exponential':
                temperature *= alpha
            elif self.cooling_schedule == 'linear':
                temperature = max(final_temp, self.initial_temp - (self.initial_temp - final_temp) * (time.time() - start_time) / self.time_limit)
            else:  # geometric
                temperature *= alpha
                
        return best_tour

    def _generate_neighbor(self, tour: List[int], dist: np.ndarray, iteration: int) -> List[int]:
        """Generate neighbor using multiple operators"""
        n = len(tour)
        if n < 4:
            return None
            
        # Choose operator based on iteration and problem size
        if iteration % 3 == 0:
            return self._two_opt_move(tour, dist)
        elif iteration % 3 == 1:
            return self._three_opt_move(tour, dist)
        else:
            return self._or_opt_move(tour, dist)

    def _two_opt_move(self, tour: List[int], dist: np.ndarray) -> List[int]:
        """2-opt move"""
        n = len(tour)
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)
        
        new_tour = tour[:]
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour

    def _three_opt_move(self, tour: List[int], dist: np.ndarray) -> List[int]:
        """3-opt move"""
        n = len(tour)
        if n < 6:
            return self._two_opt_move(tour, dist)
            
        i = random.randint(1, n - 4)
        j = random.randint(i + 1, n - 3)
        k = random.randint(j + 1, n - 1)
        
        # Try different 3-opt configurations
        configs = [
            tour[:i] + tour[j:k] + tour[i:j] + tour[k:],
            tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:],
            tour[:i] + tour[j:k] + tour[i:j][::-1] + tour[k:],
            tour[:i] + tour[j:k][::-1] + tour[i:j][::-1] + tour[k:]
        ]
        
        # Return the best configuration
        best_tour = tour
        best_cost = self._tour_cost(tour, dist)
        
        for config in configs:
            cost = self._tour_cost(config, dist)
            if cost < best_cost:
                best_tour = config
                best_cost = cost
                
        return best_tour

    def _or_opt_move(self, tour: List[int], dist: np.ndarray) -> List[int]:
        """Or-opt move (relocate a segment)"""
        n = len(tour)
        if n < 4:
            return tour
            
        # Select segment to move
        segment_length = random.randint(1, min(3, n // 3))
        start = random.randint(0, n - segment_length)
        segment = tour[start:start + segment_length]
        
        # Remove segment
        new_tour = tour[:start] + tour[start + segment_length:]
        
        # Insert segment at new position
        insert_pos = random.randint(0, len(new_tour))
        new_tour = new_tour[:insert_pos] + segment + new_tour[insert_pos:]
        
        return new_tour

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
            "name": "AdvancedSimulatedAnnealingSolver",
            "description": "Advanced Simulated Annealing (quality-focused)",
            "time_limit": self.time_limit,
            "initial_temp": self.initial_temp,
            "cooling_schedule": self.cooling_schedule
        }
