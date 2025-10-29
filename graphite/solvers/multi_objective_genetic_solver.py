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

class MultiObjectiveGeneticSolver(BaseSolver):
    """
    Multi-Objective Genetic Algorithm for TSP - quality-focused.
    
    Optimizes multiple objectives simultaneously to find
    high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, population_size: int = 80, 
                 generations: int = 1200, mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8, elite_size: int = 15):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

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

        # Evaluate initial population
        for individual in population:
            if (time.time() - start_time) >= self.time_limit:
                break
            individual['objectives'] = self._evaluate_objectives(individual['solution'], distance_matrix)
            individual['rank'] = 0
            individual['crowding_distance'] = 0.0

        # Multi-objective GA main loop
        for generation in range(self.generations):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            if self.future_tracker.get(future_id):
                return None
            
            # Non-dominated sorting
            fronts = self._non_dominated_sorting(population)
            
            # Calculate crowding distance
            for front in fronts:
                self._calculate_crowding_distance(front)
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep best individuals
            elite_count = 0
            for front in fronts:
                if elite_count >= self.elite_size:
                    break
                for individual in front:
                    if elite_count >= self.elite_size:
                        break
                    new_population.append(individual.copy())
                    elite_count += 1
            
            # Generate offspring
            while len(new_population) < self.population_size:
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                parent1 = self._tournament_selection(population, tournament_size=3)
                parent2 = self._tournament_selection(population, tournament_size=3)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1['solution'][:], parent2['solution'][:]
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, distance_matrix, start_time)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, distance_matrix, start_time)
                
                # Evaluate offspring
                child1_obj = self._evaluate_objectives(child1, distance_matrix)
                child2_obj = self._evaluate_objectives(child2, distance_matrix)
                
                new_population.extend([
                    {'solution': child1, 'objectives': child1_obj, 'rank': 0, 'crowding_distance': 0.0},
                    {'solution': child2, 'objectives': child2_obj, 'rank': 0, 'crowding_distance': 0.0}
                ])
            
            # Replace population
            population = new_population[:self.population_size]
            
            # Adaptive parameters
            if generation % 100 == 0:
                self._adapt_parameters(population, generation)
        
        # Return best solution (minimum distance)
        best_individual = min(population, key=lambda x: x['objectives'][0])
        return best_individual['solution'] + [best_individual['solution'][0]]

    def _initialize_population(self, n: int, dist: np.ndarray, start_time: float) -> List[dict]:
        """Initialize population with diverse solutions"""
        population = []
        
        # Add nearest neighbor solutions
        for _ in range(min(15, self.population_size // 5)):
            if (time.time() - start_time) >= self.time_limit:
                break
            nn_solution = self._nearest_neighbor(dist, start_time)
            if nn_solution and len(nn_solution) == n:
                population.append({'solution': nn_solution})
        
        # Add random solutions
        while len(population) < self.population_size and (time.time() - start_time) < self.time_limit:
            solution = list(range(n))
            random.shuffle(solution)
            population.append({'solution': solution})
            
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

    def _evaluate_objectives(self, solution: List[int], distance_matrix: np.ndarray) -> List[float]:
        """Evaluate multiple objectives"""
        if len(solution) < 2:
            return [float('inf'), float('inf')]
        
        # Objective 1: Total distance (minimize)
        total_distance = 0
        for i in range(len(solution) - 1):
            total_distance += distance_matrix[solution[i]][solution[i + 1]]
        
        # Objective 2: Tour balance (minimize variance of edge lengths)
        edge_lengths = []
        for i in range(len(solution) - 1):
            edge_lengths.append(distance_matrix[solution[i]][solution[i + 1]])
        
        if len(edge_lengths) > 1:
            balance = np.var(edge_lengths)
        else:
            balance = 0.0
        
        return [total_distance, balance]

    def _non_dominated_sorting(self, population: List[dict]) -> List[List[dict]]:
        """Non-dominated sorting"""
        fronts = []
        remaining = population[:]
        
        while remaining:
            current_front = []
            dominated = []
            
            for i, individual in enumerate(remaining):
                is_dominated = False
                for j, other in enumerate(remaining):
                    if i != j and self._dominates(other['objectives'], individual['objectives']):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(individual)
                else:
                    dominated.append(individual)
            
            fronts.append(current_front)
            remaining = dominated
        
        return fronts

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2"""
        at_least_one_better = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:  # obj1 is worse in this objective
                return False
            elif o1 < o2:  # obj1 is better in this objective
                at_least_one_better = True
        return at_least_one_better

    def _calculate_crowding_distance(self, front: List[dict]):
        """Calculate crowding distance for individuals in a front"""
        if len(front) <= 2:
            for individual in front:
                individual['crowding_distance'] = float('inf')
            return
        
        # Initialize crowding distance
        for individual in front:
            individual['crowding_distance'] = 0.0
        
        # Calculate for each objective
        for obj_idx in range(len(front[0]['objectives'])):
            # Sort by objective value
            front.sort(key=lambda x: x['objectives'][obj_idx])
            
            # Set boundary points
            front[0]['crowding_distance'] = float('inf')
            front[-1]['crowding_distance'] = float('inf')
            
            # Calculate range
            obj_values = [ind['objectives'][obj_idx] for ind in front]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range > 0:
                # Calculate crowding distance
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1]['objectives'][obj_idx] - 
                              front[i - 1]['objectives'][obj_idx]) / obj_range
                    front[i]['crowding_distance'] += distance

    def _tournament_selection(self, population: List[dict], tournament_size: int = 3) -> dict:
        """Tournament selection based on rank and crowding distance"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Sort by rank first, then by crowding distance
        tournament.sort(key=lambda x: (x['rank'], -x['crowding_distance']))
        
        return tournament[0]

    def _order_crossover(self, parent1: dict, parent2: dict) -> Tuple[List[int], List[int]]:
        """Order crossover (OX)"""
        solution1 = parent1['solution']
        solution2 = parent2['solution']
        n = len(solution1)
        
        i, j = sorted(random.sample(range(n), 2))
        
        # Create child1
        child1 = [-1] * n
        child1[i:j] = solution1[i:j]
        
        # Fill remaining positions from parent2
        k = j
        for city in solution2[j:] + solution2[:j]:
            if city not in child1:
                child1[k % n] = city
                k += 1
        
        # Create child2 (swap parents)
        child2 = [-1] * n
        child2[i:j] = solution2[i:j]
        
        k = j
        for city in solution1[j:] + solution1[:j]:
            if city not in child2:
                child2[k % n] = city
                k += 1
        
        return child1, child2

    def _mutate(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Advanced mutation with multiple operators"""
        n = len(solution)
        mutated = solution[:]
        
        mutation_type = random.choice(['swap', 'inversion', 'insertion', 'scramble'])
        
        if mutation_type == 'swap':
            i, j = random.sample(range(n), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        elif mutation_type == 'inversion':
            i, j = sorted(random.sample(range(n), 2))
            mutated[i:j+1] = reversed(mutated[i:j+1])
        elif mutation_type == 'insertion':
            i, j = random.sample(range(n), 2)
            if i < j:
                mutated.insert(j, mutated.pop(i))
            else:
                mutated.insert(i, mutated.pop(j))
        elif mutation_type == 'scramble':
            i, j = sorted(random.sample(range(n), 2))
            segment = mutated[i:j+1]
            random.shuffle(segment)
            mutated[i:j+1] = segment
            
        return mutated

    def _adapt_parameters(self, population: List[dict], generation: int):
        """Adapt parameters based on population state"""
        if len(population) < 2:
            return
        
        # Calculate diversity
        objectives = [ind['objectives'] for ind in population]
        diversity = np.mean([np.std([obj[i] for obj in objectives]) for i in range(len(objectives[0]))])
        
        if diversity < 1000:  # Low diversity
            self.mutation_rate = min(0.4, self.mutation_rate * 1.05)
        else:  # High diversity
            self.mutation_rate = max(0.1, self.mutation_rate * 0.98)

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "MultiObjectiveGeneticSolver",
            "description": "Multi-Objective Genetic Algorithm",
            "time_limit": self.time_limit,
            "population_size": self.population_size,
            "generations": self.generations
        }
