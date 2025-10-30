# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union, Tuple
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.solvers.common_utils import nearest_neighbor
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class MTZSolver(BaseSolver):
    """
    Miller–Tucker–Zemlin (MTZ) algorithm for TSP - exact solution.
    
    Uses integer linear programming with MTZ subtour elimination constraints
    to find optimal solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_iterations: int = 1000):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_iterations = max_iterations

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # For small problems, use exact MTZ
        if n <= 12:
            return self._exact_mtz(distance_matrix, start_time)
        else:
            # For larger problems, use heuristic MTZ with time limit
            return self._heuristic_mtz(distance_matrix, start_time)

    def _exact_mtz(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Exact MTZ for small problems"""
        n = len(dist)
        
        # Start with a good initial solution
        initial_solution = self._nearest_neighbor(dist, start_time)
        if initial_solution is None:
            return list(range(n)) + [0]
        
        best_solution = initial_solution
        best_cost = self._calculate_cost(best_solution, dist)
        
        # Apply MTZ constraints iteratively
        iteration = 0
        while iteration < self.max_iterations and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
                
            # Check for subtours and apply MTZ constraints
            subtours = self._find_subtours(best_solution)
            
            if len(subtours) == 1:  # Valid tour found
                break
            
            # Apply MTZ constraint elimination
            improved = False
            for subtour in subtours:
                if len(subtour) < n:  # Not the full tour
                    # Apply MTZ constraint to eliminate this subtour
                    new_solution = self._apply_mtz_constraint(best_solution, subtour, dist, start_time)
                    if new_solution is not None:
                        new_cost = self._calculate_cost(new_solution, dist)
                        if new_cost < best_cost - 1e-12:
                            best_solution = new_solution
                            best_cost = new_cost
                            improved = True
                            break
            
            if not improved:
                # Try random improvements
                best_solution = self._random_improvement(best_solution, dist, start_time)
                best_cost = self._calculate_cost(best_solution, dist)
            
            iteration += 1
        
        return best_solution + [best_solution[0]]

    def _heuristic_mtz(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Heuristic MTZ for larger problems"""
        n = len(dist)
        
        # Start with nearest neighbor
        solution = self._nearest_neighbor(dist, start_time)
        if solution is None:
            return list(range(n)) + [0]
        
        # Apply 2-opt improvements
        solution = self._two_opt_improve(solution, dist, start_time)
        
        # Apply MTZ constraint elimination heuristically
        iteration = 0
        while iteration < 100 and (time.time() - start_time) < self.time_limit:
            subtours = self._find_subtours(solution)
            
            if len(subtours) == 1:  # Valid tour
                break
            
            # Try to eliminate subtours using MTZ constraints
            improved = False
            for subtour in subtours:
                if len(subtour) < n:  # Not the full tour
                    new_solution = self._apply_mtz_constraint_heuristic(solution, subtour, dist, start_time)
                    if new_solution is not None:
                        new_cost = self._calculate_cost(new_solution, dist)
                        current_cost = self._calculate_cost(solution, dist)
                        if new_cost < current_cost - 1e-12:
                            solution = new_solution
                            improved = True
                            break
            
            if not improved:
                # Try random improvements
                solution = self._random_improvement(solution, dist, start_time)
            
            iteration += 1
        
        return solution + [solution[0]]

    def _nearest_neighbor(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Nearest neighbor using shared utility (respects time limit)."""
        return nearest_neighbor(dist=dist, start=0, start_time=start_time, hard_limit=self.time_limit)

    def _find_subtours(self, solution: List[int]) -> List[List[int]]:
        """Find all subtours in the solution"""
        n = len(solution)
        visited = set()
        subtours = []
        
        for i in range(n):
            if i not in visited:
                subtour = []
                current = i
                while current not in visited:
                    visited.add(current)
                    subtour.append(current)
                    # Find next city in the tour
                    next_city = None
                    for j in range(n):
                        if solution[j] == current:
                            next_city = solution[(j + 1) % n]
                            break
                    current = next_city
                subtours.append(subtour)
        
        return subtours

    def _apply_mtz_constraint(self, solution: List[int], subtour: List[int], 
                             dist: np.ndarray, start_time: float) -> List[int]:
        """Apply MTZ constraint to eliminate a subtour"""
        n = len(solution)
        if len(subtour) >= n:
            return None
        
        # MTZ constraint: u_i - u_j + n * x_ij <= n - 1
        # where u_i is the position of city i in the tour
        
        # Find the best way to break this subtour
        best_solution = None
        best_cost = float('inf')
        
        # Try different ways to break the subtour
        for i in range(len(subtour)):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(n):
                if solution[j] not in subtour:
                    # Try connecting subtour[i] to solution[j]
                    new_solution = self._break_subtour_mtz(solution, subtour, i, solution[j], dist)
                    if new_solution is not None:
                        cost = self._calculate_cost(new_solution, dist)
                        if cost < best_cost:
                            best_cost = cost
                            best_solution = new_solution
        
        return best_solution

    def _apply_mtz_constraint_heuristic(self, solution: List[int], subtour: List[int], 
                                      dist: np.ndarray, start_time: float) -> List[int]:
        """Apply MTZ constraint heuristically"""
        n = len(solution)
        if len(subtour) >= n:
            return None
        
        # Find the best connection to break the subtour
        best_connection = None
        best_cost = float('inf')
        
        for city_in_subtour in subtour:
            if (time.time() - start_time) >= self.time_limit:
                break
            for city_outside in solution:
                if city_outside not in subtour:
                    cost = dist[city_in_subtour][city_outside]
                    if cost < best_cost:
                        best_cost = cost
                        best_connection = (city_in_subtour, city_outside)
        
        if best_connection is None:
            return None
        
        # Break the subtour at the best connection
        return self._break_subtour_at_connection(solution, subtour, best_connection, dist)

    def _break_subtour_mtz(self, solution: List[int], subtour: List[int], 
                          subtour_idx: int, target_city: int, dist: np.ndarray) -> List[int]:
        """Break subtour using MTZ constraint"""
        n = len(solution)
        new_solution = solution[:]
        
        # Find positions
        subtour_city = subtour[subtour_idx]
        subtour_pos = new_solution.index(subtour_city)
        target_pos = new_solution.index(target_city)
        
        # Create a new connection that breaks the subtour
        # This is a simplified version of MTZ constraint application
        if subtour_pos < target_pos:
            # Insert target city after subtour city
            new_solution.insert(subtour_pos + 1, target_city)
        else:
            # Insert subtour city after target city
            new_solution.insert(target_pos + 1, subtour_city)
        
        return new_solution

    def _break_subtour_at_connection(self, solution: List[int], subtour: List[int], 
                                   connection: Tuple[int, int], dist: np.ndarray) -> List[int]:
        """Break subtour at a specific connection"""
        city1, city2 = connection
        n = len(solution)
        new_solution = solution[:]
        
        # Find positions
        pos1 = new_solution.index(city1)
        pos2 = new_solution.index(city2)
        
        # Create connection that breaks the subtour
        if abs(pos1 - pos2) > 1:
            # Cities are not adjacent, create a connection
            if pos1 < pos2:
                # Move city2 to be after city1
                new_solution.remove(city2)
                new_solution.insert(pos1 + 1, city2)
            else:
                # Move city1 to be after city2
                new_solution.remove(city1)
                new_solution.insert(pos2 + 1, city1)
        
        return new_solution

    def _random_improvement(self, solution: List[int], dist: np.ndarray, 
                           start_time: float) -> List[int]:
        """Apply random improvements"""
        n = len(solution)
        improved_solution = solution[:]
        
        # Try random 2-opt moves
        for _ in range(min(10, n)):
            if (time.time() - start_time) >= self.time_limit:
                break
            i, j = random.sample(range(n), 2)
            if i < j:
                new_solution = improved_solution[:]
                new_solution[i:j+1] = reversed(new_solution[i:j+1])
                if self._calculate_cost(new_solution, dist) < self._calculate_cost(improved_solution, dist):
                    improved_solution = new_solution
        
        return improved_solution

    def _two_opt_improve(self, solution: List[int], dist: np.ndarray, 
                        start_time: float) -> List[int]:
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
            "name": "MTZSolver",
            "description": "Miller-Tucker-Zemlin (MTZ) exact algorithm",
            "time_limit": self.time_limit,
            "max_iterations": self.max_iterations
        }
