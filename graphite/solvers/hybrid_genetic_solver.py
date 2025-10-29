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

class HybridGeneticSolver(BaseSolver):
    """
    Hybrid Genetic Algorithm for TSP - quality-focused.
    
    Combines multiple genetic operators with various local search
    strategies to find high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, population_size: int = 100, 
                 generations: int = 1500, mutation_rate: float = 0.15,
                 crossover_rate: float = 0.9, elite_size: int = 20,
                 local_search_rate: float = 0.6, diversity_threshold: float = 0.1):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.local_search_rate = local_search_rate
        self.diversity_threshold = diversity_threshold

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # Initialize population with diverse strategies
        population = self._initialize_diverse_population(n, distance_matrix, start_time)
        if not population:
            return list(range(n)) + [0]

        # Evaluate initial population
        fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
        
        # Hybrid genetic algorithm main loop
        stagnation_counter = 0
        best_fitness = min(fitness_scores)
        
        for generation in range(self.generations):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            if self.future_tracker.get(future_id):
                return None
            
            # Check for stagnation
            current_best = min(fitness_scores)
            if current_best < best_fitness - 1e-6:
                best_fitness = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adaptive restart if stagnated
            if stagnation_counter > 50:
                population = self._adaptive_restart(population, fitness_scores, n, distance_matrix, start_time)
                fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
                stagnation_counter = 0
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
            
            # Generate offspring with multiple strategies
            while len(new_population) < self.population_size:
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                # Choose crossover strategy based on diversity
                diversity = self._calculate_diversity(population, fitness_scores)
                if diversity < self.diversity_threshold:
                    # Low diversity: use disruptive crossover
                    parent1 = self._tournament_selection(population, fitness_scores, tournament_size=5)
                    parent2 = self._tournament_selection(population, fitness_scores, tournament_size=5)
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    # High diversity: use conservative crossover
                    parent1 = self._tournament_selection(population, fitness_scores, tournament_size=3)
                    parent2 = self._tournament_selection(population, fitness_scores, tournament_size=3)
                    child1, child2 = self._edge_recombination_crossover(parent1, parent2)
                
                # Adaptive mutation
                mutation_rate = self.mutation_rate
                if diversity < self.diversity_threshold:
                    mutation_rate *= 2.0  # Increase mutation for low diversity
                
                # Mutation
                if random.random() < mutation_rate:
                    child1 = self._adaptive_mutate(child1, distance_matrix, generation)
                if random.random() < mutation_rate:
                    child2 = self._adaptive_mutate(child2, distance_matrix, generation)
                
                # Local search with multiple strategies
                if random.random() < self.local_search_rate:
                    child1 = self._multi_strategy_local_search(child1, distance_matrix, start_time)
                if random.random() < self.local_search_rate:
                    child2 = self._multi_strategy_local_search(child2, distance_matrix, start_time)
                
                new_population.extend([child1, child2])
            
            # Replace population
            population = new_population[:self.population_size]
            
            # Evaluate new population
            fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
            
            # Adaptive parameters
            if generation % 100 == 0:
                self._adapt_parameters(fitness_scores, generation, diversity)
        
        # Return best individual
        best_idx = np.argmin(fitness_scores)
        best_tour = population[best_idx]
        return best_tour + [best_tour[0]]

    def _initialize_diverse_population(self, n: int, dist: np.ndarray, start_time: float) -> List[List[int]]:
        """Initialize population with diverse strategies"""
        population = []
        
        # Nearest neighbor variants
        for _ in range(min(15, self.population_size // 6)):
            if (time.time() - start_time) >= self.time_limit:
                break
            nn_tour = self._simple_nn(dist)
            if nn_tour and len(nn_tour) == n:
                population.append(nn_tour)
        
        # Random solutions
        for _ in range(min(20, self.population_size // 4)):
            if (time.time() - start_time) >= self.time_limit:
                break
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
        
        # Greedy solutions with different starting points
        for start in range(0, min(n, 10)):
            if (time.time() - start_time) >= self.time_limit:
                break
            greedy_tour = self._greedy_from_start(dist, start)
            if greedy_tour and len(greedy_tour) == n:
                population.append(greedy_tour)
        
        # Fill remaining with random
        while len(population) < self.population_size and (time.time() - start_time) < self.time_limit:
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
            
        return population

    def _simple_nn(self, dist: np.ndarray) -> List[int]:
        """Simple nearest neighbor implementation"""
        n = len(dist)
        tour = [0]
        visited = {0}
        current = 0
        for _ in range(n - 1):
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

    def _greedy_from_start(self, dist: np.ndarray, start: int) -> List[int]:
        """Greedy solution starting from specific city"""
        n = len(dist)
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n - 1):
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

    def _calculate_fitness(self, individual: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate fitness (tour distance)"""
        total_distance = 0
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i]][individual[i + 1]]
        return total_distance

    def _calculate_diversity(self, population: List[List[int]], fitness_scores: List[float]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 1.0
        
        # Calculate average pairwise distance
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._tour_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 1.0

    def _tour_distance(self, tour1: List[int], tour2: List[int]) -> float:
        """Calculate distance between two tours"""
        n = len(tour1)
        distance = 0
        for i in range(n):
            if tour1[i] != tour2[i]:
                distance += 1
        return distance / n

    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float], tournament_size: int = 3) -> List[int]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx][:]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX)"""
        n = len(parent1)
        i, j = sorted(random.sample(range(n), 2))
        
        # Create child1
        child1 = [-1] * n
        child1[i:j] = parent1[i:j]
        
        # Fill remaining positions from parent2
        k = j
        for city in parent2[j:] + parent2[:j]:
            if city not in child1:
                child1[k % n] = city
                k += 1
        
        # Create child2 (swap parents)
        child2 = [-1] * n
        child2[i:j] = parent2[i:j]
        
        k = j
        for city in parent1[j:] + parent1[:j]:
            if city not in child2:
                child2[k % n] = city
                k += 1
        
        return child1, child2

    def _edge_recombination_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Edge recombination crossover (ERX)"""
        n = len(parent1)
        
        # Build edge maps
        edge_map1 = self._build_edge_map(parent1)
        edge_map2 = self._build_edge_map(parent2)
        
        # Create child1
        child1 = self._build_child_from_edge_maps(edge_map1, edge_map2, n)
        
        # Create child2 (swap parents)
        child2 = self._build_child_from_edge_maps(edge_map2, edge_map1, n)
        
        return child1, child2

    def _build_edge_map(self, tour: List[int]) -> dict:
        """Build edge map for a tour"""
        n = len(tour)
        edge_map = {}
        
        for i in range(n):
            city = tour[i]
            prev_city = tour[i - 1]
            next_city = tour[(i + 1) % n]
            
            if city not in edge_map:
                edge_map[city] = set()
            edge_map[city].add(prev_city)
            edge_map[city].add(next_city)
            
        return edge_map

    def _build_child_from_edge_maps(self, edge_map1: dict, edge_map2: dict, n: int) -> List[int]:
        """Build child from edge maps"""
        child = []
        remaining = set(range(n))
        current = random.randint(0, n - 1)
        child.append(current)
        remaining.remove(current)
        
        while remaining:
            # Find cities connected to current city
            connections = set()
            if current in edge_map1:
                connections.update(edge_map1[current])
            if current in edge_map2:
                connections.update(edge_map2[current])
            
            # Filter to remaining cities
            available = connections.intersection(remaining)
            
            if available:
                # Choose city with fewest remaining connections
                next_city = min(available, key=lambda city: len(remaining.intersection(
                    edge_map1.get(city, set()).union(edge_map2.get(city, set()))
                )))
            else:
                # Choose random remaining city
                next_city = random.choice(list(remaining))
            
            child.append(next_city)
            remaining.remove(next_city)
            current = next_city
            
        return child

    def _adaptive_mutate(self, individual: List[int], dist: np.ndarray, generation: int) -> List[int]:
        """Adaptive mutation with multiple operators"""
        mutated = individual[:]
        
        # Choose mutation type based on generation
        if generation < 500:
            mutation_type = random.choice(['swap', 'inversion'])
        else:
            mutation_type = random.choice(['swap', 'inversion', 'insertion', 'scramble'])
        
        if mutation_type == 'swap':
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        elif mutation_type == 'inversion':
            i, j = sorted(random.sample(range(len(mutated)), 2))
            mutated[i:j+1] = reversed(mutated[i:j+1])
        elif mutation_type == 'insertion':
            i, j = random.sample(range(len(mutated)), 2)
            if i < j:
                mutated.insert(j, mutated.pop(i))
            else:
                mutated.insert(i, mutated.pop(j))
        elif mutation_type == 'scramble':
            i, j = sorted(random.sample(range(len(mutated)), 2))
            segment = mutated[i:j+1]
            random.shuffle(segment)
            mutated[i:j+1] = segment
            
        return mutated

    def _multi_strategy_local_search(self, individual: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Multi-strategy local search"""
        n = len(individual)
        current_tour = individual[:]
        
        # Try different local search strategies
        strategies = ['2opt', '3opt', 'or_opt']
        
        for strategy in strategies:
            if (time.time() - start_time) >= self.time_limit:
                break
                
            if strategy == '2opt':
                improved = self._two_opt_improve(current_tour, dist, start_time)
            elif strategy == '3opt':
                improved = self._three_opt_improve(current_tour, dist, start_time)
            elif strategy == 'or_opt':
                improved = self._or_opt_improve(current_tour, dist, start_time)
            
            if improved:
                break  # Stop at first improvement
                
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
                        if self._calculate_fitness(config, dist) < self._calculate_fitness(tour, dist) - 1e-12:
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
                    if self._calculate_fitness(new_tour, dist) < self._calculate_fitness(tour, dist) - 1e-12:
                        tour[:] = new_tour
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    def _adaptive_restart(self, population: List[List[int]], fitness_scores: List[float], 
                         n: int, dist: np.ndarray, start_time: float) -> List[List[int]]:
        """Adaptive restart to maintain diversity"""
        # Keep best 20% of population
        elite_size = max(1, len(population) // 5)
        elite_indices = np.argsort(fitness_scores)[:elite_size]
        new_population = [population[idx][:] for idx in elite_indices]
        
        # Add some random solutions
        while len(new_population) < len(population) and (time.time() - start_time) < self.time_limit:
            tour = list(range(n))
            random.shuffle(tour)
            new_population.append(tour)
            
        return new_population

    def _adapt_parameters(self, fitness_scores: List[float], generation: int, diversity: float):
        """Adapt parameters based on population state"""
        progress = generation / self.generations
        
        if diversity < self.diversity_threshold:
            # Low diversity: increase exploration
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            self.local_search_rate = min(0.8, self.local_search_rate * 1.05)
        else:
            # High diversity: maintain balance
            self.mutation_rate = max(0.05, self.mutation_rate * 0.98)
            self.local_search_rate = max(0.3, self.local_search_rate * 0.99)

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "HybridGeneticSolver",
            "description": "Hybrid Genetic Algorithm (multi-strategy)",
            "time_limit": self.time_limit,
            "population_size": self.population_size,
            "generations": self.generations
        }
