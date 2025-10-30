# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.solvers.common_utils import two_opt_improve
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class IteratedLocalSearchSolver(BaseSolver):
    """
    Iterated Local Search for TSP - quality-focused.
    
    Combines local search with perturbation to escape local optima
    and find high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_iterations: int = 1000,
                 perturbation_strength: float = 0.3, acceptance_criterion: str = 'better'):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.perturbation_strength = perturbation_strength
        self.acceptance_criterion = acceptance_criterion

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

        # Apply local search to initial solution
        current_solution = self._local_search(current_solution, distance_matrix, start_time)
        best_solution = current_solution[:]
        best_cost = self._calculate_cost(best_solution, distance_matrix)

        # Iterated Local Search main loop
        iteration = 0
        no_improvement_count = 0
        
        while iteration < self.max_iterations and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
            
            # Perturbation
            perturbed_solution = self._perturb(current_solution, distance_matrix, start_time)
            if perturbed_solution is None:
                break
            
            # Local search on perturbed solution
            improved_solution = self._local_search(perturbed_solution, distance_matrix, start_time)
            improved_cost = self._calculate_cost(improved_solution, distance_matrix)
            current_cost = self._calculate_cost(current_solution, distance_matrix)
            
            # Acceptance criterion
            if self._accept_solution(improved_cost, current_cost, best_cost, iteration):
                current_solution = improved_solution
                current_cost = improved_cost
                
                if improved_cost < best_cost - 1e-12:
                    best_solution = improved_solution[:]
                    best_cost = improved_cost
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Adaptive perturbation
            if no_improvement_count > 50:
                self._adapt_perturbation_strength()
                no_improvement_count = 0
            
            iteration += 1
        
        return best_solution + [best_solution[0]]

    def _generate_initial_solution(self, n: int, dist: np.ndarray, start_time: float) -> List[int]:
        """Generate initial solution using multiple strategies"""
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
        """Greedy construction with random elements"""
        n = len(dist)
        tour = []
        remaining = set(range(n))
        
        # Start with random city
        current = random.choice(list(remaining))
        tour.append(current)
        remaining.remove(current)
        
        while remaining and (time.time() - start_time) < self.time_limit:
            # Choose next city with some randomness
            if random.random() < 0.7:  # 70% greedy, 30% random
                nearest = None
                best_dist = float('inf')
                for j in remaining:
                    if dist[current][j] < best_dist:
                        best_dist = dist[current][j]
                        nearest = j
                if nearest is not None:
                    current = nearest
            else:
                current = random.choice(list(remaining))
            
            tour.append(current)
            remaining.remove(current)
        
        return tour

    def _local_search(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Multi-strategy local search"""
        n = len(solution)
        current_solution = solution[:]
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations and (time.time() - start_time) < self.time_limit:
            improved = False
            iterations += 1
            
            # Try different local search strategies
            strategies = ['2opt', '3opt', 'or_opt']
            
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
        
        return current_solution

    def _two_opt_improve(self, tour: List[int], dist: np.ndarray, start_time: float) -> bool:
        """2-opt local improvement via shared utility; return whether improved"""
        improved_tour = two_opt_improve(solution=tour, dist=dist, start_time=start_time, hard_limit=self.time_limit)
        if improved_tour != tour:
            tour[:] = improved_tour
            return True
        return False

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

    def _perturb(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Perturbation operator"""
        n = len(solution)
        perturbed = solution[:]
        
        # Apply multiple perturbation moves
        num_moves = max(1, int(n * self.perturbation_strength))
        
        for _ in range(num_moves):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            move_type = random.choice(['swap', 'insertion', 'inversion', 'scramble'])
            
            if move_type == 'swap':
                i, j = random.sample(range(n), 2)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            elif move_type == 'insertion':
                i, j = random.sample(range(n), 2)
                if i < j:
                    perturbed.insert(j, perturbed.pop(i))
                else:
                    perturbed.insert(i, perturbed.pop(j))
            elif move_type == 'inversion':
                i, j = sorted(random.sample(range(n), 2))
                perturbed[i:j+1] = reversed(perturbed[i:j+1])
            elif move_type == 'scramble':
                i, j = sorted(random.sample(range(n), 2))
                segment = perturbed[i:j+1]
                random.shuffle(segment)
                perturbed[i:j+1] = segment
        
        return perturbed

    def _accept_solution(self, new_cost: float, current_cost: float, best_cost: float, iteration: int) -> bool:
        """Acceptance criterion"""
        if self.acceptance_criterion == 'better':
            return new_cost < current_cost - 1e-12
        elif self.acceptance_criterion == 'best':
            return new_cost < best_cost - 1e-12
        elif self.acceptance_criterion == 'simulated_annealing':
            # Simulated annealing acceptance
            if new_cost < current_cost - 1e-12:
                return True
            else:
                temperature = max(0.1, 1.0 - iteration / self.max_iterations)
                probability = np.exp(-(new_cost - current_cost) / temperature)
                return random.random() < probability
        else:
            return new_cost < current_cost - 1e-12

    def _adapt_perturbation_strength(self):
        """Adapt perturbation strength"""
        if self.perturbation_strength < 0.5:
            self.perturbation_strength *= 1.1
        else:
            self.perturbation_strength = 0.3

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
            "name": "IteratedLocalSearchSolver",
            "description": "Iterated Local Search (quality-focused)",
            "time_limit": self.time_limit,
            "max_iterations": self.max_iterations,
            "perturbation_strength": self.perturbation_strength
        }
