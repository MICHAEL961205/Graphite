# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union, Set, Tuple
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class TabuSearchSolver(BaseSolver):
    """
    Tabu Search for TSP - quality-focused.
    
    Uses tabu list to prevent cycling and escape local optima
    to find high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_iterations: int = 2000,
                 tabu_tenure: int = 20, aspiration_criteria: bool = True):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.aspiration_criteria = aspiration_criteria

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # Generate initial solution
        current_solution = self._generate_initial_solution(n, distance_matrix, start_time)
        if current_solution is None:
            return list(range(n)) + [0]

        best_solution = current_solution[:]
        best_cost = self._calculate_cost(best_solution, distance_matrix)
        
        # Tabu search components
        tabu_list = []
        tabu_tenure = self.tabu_tenure
        no_improvement_count = 0
        
        # Tabu search main loop
        iteration = 0
        while iteration < self.max_iterations and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
            
            # Generate neighborhood
            neighborhood = self._generate_neighborhood(current_solution, distance_matrix, start_time)
            if not neighborhood:
                break
            
            # Find best non-tabu move
            best_move = None
            best_cost_change = float('inf')
            
            for move, new_solution, cost_change in neighborhood:
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                # Check if move is tabu
                is_tabu = self._is_tabu_move(move, tabu_list)
                
                # Aspiration criteria
                if is_tabu and self.aspiration_criteria:
                    new_cost = self._calculate_cost(new_solution, distance_matrix)
                    if new_cost < best_cost - 1e-12:
                        is_tabu = False  # Override tabu status
                
                if not is_tabu and cost_change < best_cost_change:
                    best_cost_change = cost_change
                    best_move = (move, new_solution, cost_change)
            
            # Apply best move
            if best_move is not None:
                move, new_solution, cost_change = best_move
                current_solution = new_solution
                current_cost = self._calculate_cost(current_solution, distance_matrix)
                
                # Update tabu list
                tabu_list.append((move, iteration + tabu_tenure))
                
                # Remove expired tabu moves
                tabu_list = [(m, t) for m, t in tabu_list if t > iteration]
                
                # Update best solution
                if current_cost < best_cost - 1e-12:
                    best_solution = current_solution[:]
                    best_cost = current_cost
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Adaptive tabu tenure
                if no_improvement_count > 100:
                    tabu_tenure = min(50, tabu_tenure + 5)
                    no_improvement_count = 0
                elif no_improvement_count < 20:
                    tabu_tenure = max(5, tabu_tenure - 1)
            else:
                # No valid moves, diversify
                current_solution = self._diversify_solution(current_solution, distance_matrix, start_time)
                no_improvement_count += 1
            
            iteration += 1
        
        return best_solution + [best_solution[0]]

    def _generate_initial_solution(self, n: int, dist: np.ndarray, start_time: float) -> List[int]:
        """Generate initial solution"""
        strategies = ['nearest_neighbor', 'random', 'greedy']
        
        for strategy in strategies:
            if (time.time() - start_time) >= self.time_limit:
                break
                
            if strategy == 'nearest_neighbor':
                solution = self._nearest_neighbor(dist, start_time)
            elif strategy == 'random':
                solution = list(range(n))
                random.shuffle(solution)
            elif strategy == 'greedy':
                solution = self._greedy_construction(dist, start_time)
            
            if solution and len(solution) == n:
                return solution
        
        return list(range(n))

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

    def _greedy_construction(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Greedy construction"""
        n = len(dist)
        tour = []
        remaining = set(range(n))
        
        # Start with random city
        current = random.choice(list(remaining))
        tour.append(current)
        remaining.remove(current)
        
        while remaining and (time.time() - start_time) < self.time_limit:
            nearest = None
            best_dist = float('inf')
            for j in remaining:
                if dist[current][j] < best_dist:
                    best_dist = dist[current][j]
                    nearest = j
            if nearest is None:
                break
            tour.append(nearest)
            remaining.remove(nearest)
            current = nearest
        return tour

    def _generate_neighborhood(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[Tuple]:
        """Generate neighborhood moves"""
        n = len(solution)
        neighborhood = []
        
        # 2-opt moves
        for i in range(1, n - 2):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= self.time_limit:
                    break
                new_solution = solution[:]
                new_solution[i:j] = reversed(new_solution[i:j])
                cost_change = self._calculate_move_cost(solution, new_solution, dist, i, j)
                move = ('2opt', i, j)
                neighborhood.append((move, new_solution, cost_change))
        
        # Swap moves
        for i in range(n - 1):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= self.time_limit:
                    break
                new_solution = solution[:]
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                cost_change = self._calculate_swap_cost(solution, new_solution, dist, i, j)
                move = ('swap', i, j)
                neighborhood.append((move, new_solution, cost_change))
        
        # Insertion moves
        for i in range(n):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(n):
                if (time.time() - start_time) >= self.time_limit:
                    break
                if i != j and i != (j + 1) % n:
                    new_solution = solution[:]
                    city = new_solution.pop(i)
                    new_solution.insert(j, city)
                    cost_change = self._calculate_insertion_cost(solution, new_solution, dist, i, j)
                    move = ('insertion', i, j)
                    neighborhood.append((move, new_solution, cost_change))
        
        return neighborhood

    def _calculate_move_cost(self, old_solution: List[int], new_solution: List[int], 
                           dist: np.ndarray, i: int, j: int) -> float:
        """Calculate cost change for 2-opt move"""
        n = len(old_solution)
        a, b = old_solution[i-1], old_solution[i]
        c, d = old_solution[j-1], old_solution[j]
        
        old_cost = dist[a][b] + dist[c][d]
        new_cost = dist[a][c] + dist[b][d]
        
        return new_cost - old_cost

    def _calculate_swap_cost(self, old_solution: List[int], new_solution: List[int], 
                           dist: np.ndarray, i: int, j: int) -> float:
        """Calculate cost change for swap move"""
        n = len(old_solution)
        cost_change = 0.0
        
        # Calculate cost change around position i
        if i > 0:
            old_cost = dist[old_solution[i-1]][old_solution[i]]
            new_cost = dist[new_solution[i-1]][new_solution[i]]
            cost_change += new_cost - old_cost
        
        if i < n - 1:
            old_cost = dist[old_solution[i]][old_solution[i+1]]
            new_cost = dist[new_solution[i]][new_solution[i+1]]
            cost_change += new_cost - old_cost
        
        # Calculate cost change around position j
        if j > 0:
            old_cost = dist[old_solution[j-1]][old_solution[j]]
            new_cost = dist[new_solution[j-1]][new_solution[j]]
            cost_change += new_cost - old_cost
        
        if j < n - 1:
            old_cost = dist[old_solution[j]][old_solution[j+1]]
            new_cost = dist[new_solution[j]][new_solution[j+1]]
            cost_change += new_cost - old_cost
        
        return cost_change

    def _calculate_insertion_cost(self, old_solution: List[int], new_solution: List[int], 
                                dist: np.ndarray, i: int, j: int) -> float:
        """Calculate cost change for insertion move"""
        n = len(old_solution)
        cost_change = 0.0
        
        # Remove city from position i
        if i > 0 and i < n - 1:
            old_cost = dist[old_solution[i-1]][old_solution[i]] + dist[old_solution[i]][old_solution[i+1]]
            new_cost = dist[old_solution[i-1]][old_solution[i+1]]
            cost_change += new_cost - old_cost
        
        # Insert city at position j
        if j > 0 and j < n:
            old_cost = dist[new_solution[j-1]][new_solution[j]]
            new_cost = dist[new_solution[j-1]][old_solution[i]] + dist[old_solution[i]][new_solution[j]]
            cost_change += new_cost - old_cost
        
        return cost_change

    def _is_tabu_move(self, move: Tuple, tabu_list: List[Tuple]) -> bool:
        """Check if move is tabu"""
        for tabu_move, _ in tabu_list:
            if move == tabu_move:
                return True
        return False

    def _diversify_solution(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Diversify solution when stuck"""
        n = len(solution)
        diversified = solution[:]
        
        # Apply random perturbations
        num_perturbations = min(5, n // 4)
        for _ in range(num_perturbations):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            move_type = random.choice(['swap', 'insertion', 'inversion'])
            
            if move_type == 'swap':
                i, j = random.sample(range(n), 2)
                diversified[i], diversified[j] = diversified[j], diversified[i]
            elif move_type == 'insertion':
                i, j = random.sample(range(n), 2)
                if i != j:
                    city = diversified.pop(i)
                    diversified.insert(j, city)
            elif move_type == 'inversion':
                i, j = sorted(random.sample(range(n), 2))
                diversified[i:j+1] = reversed(diversified[i:j+1])
        
        return diversified

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
            "name": "TabuSearchSolver",
            "description": "Tabu Search (quality-focused)",
            "time_limit": self.time_limit,
            "max_iterations": self.max_iterations,
            "tabu_tenure": self.tabu_tenure
        }
