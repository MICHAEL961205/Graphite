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

class EvolutionStrategiesSolver(BaseSolver):
    """
    Evolution Strategies for TSP - quality-focused.
    
    Uses self-adaptive mutation and selection to find
    high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, population_size: int = 60, 
                 generations: int = 1500, mu: int = 20, lambda_: int = 40,
                 tau: float = 1.0, tau_prime: float = 1.0):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.population_size = population_size
        self.generations = generations
        self.mu = mu  # Parent population size
        self.lambda_ = lambda_  # Offspring population size
        self.tau = tau  # Learning rate for strategy parameters
        self.tau_prime = tau_prime  # Learning rate for strategy parameters

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # Initialize population
        population = self._initialize_population(n, distance_matrix, start_time)
        if not population:
            return list(range(n)) + [0]

        # Evolution Strategies main loop
        for generation in range(self.generations):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            if self.future_tracker.get(future_id):
                return None
            
            # Generate offspring
            offspring = []
            for _ in range(self.lambda_):
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                # Select parent
                parent = self._tournament_selection(population, tournament_size=3)
                
                # Create offspring
                child = self._create_offspring(parent, n, distance_matrix, start_time)
                if child is not None:
                    offspring.append(child)
            
            # Evaluate offspring
            for individual in offspring:
                if (time.time() - start_time) >= self.time_limit:
                    break
                individual['fitness'] = self._calculate_fitness(individual['solution'], distance_matrix)
            
            # Combine parent and offspring populations
            combined_population = population + offspring
            
            # Select next generation (mu + lambda strategy)
            population = self._environmental_selection(combined_population, self.mu)
            
            # Adaptive parameters
            if generation % 100 == 0:
                self._adapt_parameters(population, generation)
        
        # Return best individual
        best_individual = min(population, key=lambda x: x['fitness'])
        return best_individual['solution'] + [best_individual['solution'][0]]

    def _initialize_population(self, n: int, dist: np.ndarray, start_time: float) -> List[dict]:
        """Initialize population with diverse solutions"""
        population = []
        
        # Add nearest neighbor solutions
        for _ in range(min(10, self.population_size // 4)):
            if (time.time() - start_time) >= self.time_limit:
                break
            nn_solution = self._nearest_neighbor(dist, start_time)
            if nn_solution and len(nn_solution) == n:
                individual = self._create_individual(nn_solution, n)
                population.append(individual)
        
        # Add random solutions
        while len(population) < self.population_size and (time.time() - start_time) < self.time_limit:
            solution = list(range(n))
            random.shuffle(solution)
            individual = self._create_individual(solution, n)
            population.append(individual)
            
        return population

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

    def _create_individual(self, solution: List[int], n: int) -> dict:
        """Create individual with strategy parameters"""
        return {
            'solution': solution[:],
            'fitness': 0.0,
            'sigma': np.random.uniform(0.1, 1.0, n),  # Strategy parameters
            'tau': self.tau,
            'tau_prime': self.tau_prime
        }

    def _create_offspring(self, parent: dict, n: int, dist: np.ndarray, start_time: float) -> dict:
        """Create offspring through mutation"""
        if (time.time() - start_time) >= self.time_limit:
            return None
            
        # Copy parent
        child = {
            'solution': parent['solution'][:],
            'fitness': 0.0,
            'sigma': parent['sigma'].copy(),
            'tau': parent['tau'],
            'tau_prime': parent['tau_prime']
        }
        
        # Mutate strategy parameters
        child['sigma'] = self._mutate_strategy_parameters(child['sigma'], n)
        
        # Mutate solution
        child['solution'] = self._mutate_solution(child['solution'], child['sigma'], dist, start_time)
        
        return child

    def _mutate_strategy_parameters(self, sigma: np.ndarray, n: int) -> np.ndarray:
        """Mutate strategy parameters"""
        # Global mutation
        tau_prime_factor = np.exp(self.tau_prime * np.random.normal(0, 1))
        sigma *= tau_prime_factor
        
        # Individual mutations
        for i in range(n):
            tau_factor = np.exp(self.tau * np.random.normal(0, 1))
            sigma[i] *= tau_factor
        
        # Keep sigma in reasonable bounds
        sigma = np.clip(sigma, 0.01, 10.0)
        
        return sigma

    def _mutate_solution(self, solution: List[int], sigma: np.ndarray, dist: np.ndarray, start_time: float) -> List[int]:
        """Mutate solution using strategy parameters"""
        n = len(solution)
        mutated = solution[:]
        
        # Apply multiple mutation operators based on sigma values
        for i in range(n):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            # Higher sigma means more aggressive mutation
            if sigma[i] > 0.5:
                mutation_type = random.choice(['swap', 'insertion', 'inversion'])
            else:
                mutation_type = random.choice(['swap', 'insertion'])
            
            if mutation_type == 'swap':
                j = random.randint(0, n - 1)
                if i != j:
                    mutated[i], mutated[j] = mutated[j], mutated[i]
            elif mutation_type == 'insertion':
                j = random.randint(0, n - 1)
                if i != j:
                    city = mutated.pop(i)
                    mutated.insert(j, city)
            elif mutation_type == 'inversion':
                j = random.randint(i + 1, n)
                mutated[i:j] = reversed(mutated[i:j])
        
        return mutated

    def _tournament_selection(self, population: List[dict], tournament_size: int = 3) -> dict:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda x: x['fitness'])

    def _environmental_selection(self, population: List[dict], mu: int) -> List[dict]:
        """Environmental selection (mu + lambda)"""
        # Sort by fitness
        population.sort(key=lambda x: x['fitness'])
        return population[:mu]

    def _calculate_fitness(self, solution: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate fitness (tour distance)"""
        if len(solution) < 2:
            return float('inf')
        total_distance = 0
        for i in range(len(solution) - 1):
            total_distance += distance_matrix[solution[i]][solution[i + 1]]
        return total_distance

    def _adapt_parameters(self, population: List[dict], generation: int):
        """Adapt parameters based on population diversity"""
        if len(population) < 2:
            return
        
        # Calculate diversity
        fitness_values = [ind['fitness'] for ind in population]
        diversity = np.std(fitness_values)
        
        # Adapt tau and tau_prime based on diversity
        if diversity < 1000:  # Low diversity
            self.tau = min(2.0, self.tau * 1.1)
            self.tau_prime = min(2.0, self.tau_prime * 1.05)
        else:  # High diversity
            self.tau = max(0.1, self.tau * 0.95)
            self.tau_prime = max(0.1, self.tau_prime * 0.98)

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "EvolutionStrategiesSolver",
            "description": "Evolution Strategies (quality-focused)",
            "time_limit": self.time_limit,
            "population_size": self.population_size,
            "generations": self.generations
        }
