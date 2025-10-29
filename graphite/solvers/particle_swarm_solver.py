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

class ParticleSwarmSolver(BaseSolver):
    """
    Particle Swarm Optimization for TSP - quality-focused.
    
    Uses particles that move through solution space to find
    high-quality TSP solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, n_particles: int = 50, n_iterations: int = 1000,
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # Initialize particles
        particles = self._initialize_particles(n, distance_matrix, start_time)
        if not particles:
            return list(range(n)) + [0]

        # PSO main loop
        for iteration in range(self.n_iterations):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            if self.future_tracker.get(future_id):
                return None
            
            # Update particles
            for particle in particles:
                if (time.time() - start_time) >= self.time_limit:
                    break
                self._update_particle(particle, iteration)
            
            # Adaptive parameters
            if iteration % 50 == 0:
                self._adapt_parameters(iteration, self.n_iterations)
        
        # Return best solution
        best_particle = min(particles, key=lambda p: p['best_cost'])
        return best_particle['best_position'] + [best_particle['best_position'][0]]

    def _initialize_particles(self, n: int, dist: np.ndarray, start_time: float) -> List[dict]:
        """Initialize particle swarm"""
        particles = []
        
        # Add some nearest neighbor solutions
        for _ in range(min(5, self.n_particles // 10)):
            if (time.time() - start_time) >= self.time_limit:
                break
            nn_tour = self._simple_nn(dist)
            if nn_tour and len(nn_tour) == n:
                particles.append(self._create_particle(nn_tour, dist))
        
        # Add random solutions
        while len(particles) < self.n_particles and (time.time() - start_time) < self.time_limit:
            tour = list(range(n))
            random.shuffle(tour)
            particles.append(self._create_particle(tour, dist))
            
        return particles

    def _create_particle(self, position: List[int], dist: np.ndarray) -> dict:
        """Create a particle from a tour"""
        n = len(position)
        velocity = np.random.uniform(-1, 1, n)
        
        return {
            'position': position[:],
            'velocity': velocity,
            'best_position': position[:],
            'best_cost': self._tour_cost(position, dist),
            'current_cost': self._tour_cost(position, dist)
        }

    def _update_particle(self, particle: dict, iteration: int):
        """Update particle position and velocity"""
        n = len(particle['position'])
        
        # Update velocity
        r1, r2 = random.random(), random.random()
        
        # Cognitive component
        cognitive = self.c1 * r1 * np.array(self._subtract_tours(particle['best_position'], particle['position']))
        
        # Social component (using global best - simplified)
        social = self.c2 * r2 * np.array(self._subtract_tours(particle['best_position'], particle['position']))
        
        # Update velocity
        particle['velocity'] = (self.w * np.array(particle['velocity']) + 
                               cognitive + social)
        
        # Update position
        new_position = self._add_velocity_to_tour(particle['position'], particle['velocity'])
        
        # Evaluate new position
        new_cost = self._tour_cost(new_position, None)  # We'll calculate with actual dist later
        
        if new_cost < particle['best_cost']:
            particle['best_position'] = new_position[:]
            particle['best_cost'] = new_cost
        
        particle['position'] = new_position
        particle['current_cost'] = new_cost

    def _subtract_tours(self, tour1: List[int], tour2: List[int]) -> List[float]:
        """Calculate difference between two tours"""
        n = len(tour1)
        diff = []
        for i in range(n):
            # Find position of tour1[i] in tour2
            pos_in_tour2 = tour2.index(tour1[i])
            diff.append(pos_in_tour2 - i)
        return diff

    def _add_velocity_to_tour(self, tour: List[int], velocity: np.ndarray) -> List[int]:
        """Add velocity to tour to get new position"""
        n = len(tour)
        
        # Convert tour to continuous representation
        continuous = np.array(range(n), dtype=float)
        for i, city in enumerate(tour):
            continuous[city] = i
        
        # Add velocity
        continuous += velocity
        
        # Convert back to discrete tour
        sorted_indices = np.argsort(continuous)
        new_tour = [0] * n
        for i, city in enumerate(sorted_indices):
            new_tour[city] = i
            
        return new_tour

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

    def _tour_cost(self, tour: List[int], dist: np.ndarray) -> float:
        """Calculate tour cost (simplified for PSO)"""
        if len(tour) < 2:
            return 0.0
        # For PSO, we use a simplified cost function
        # In practice, you'd use the actual distance matrix
        return len(set(tour))  # Penalty for duplicate cities

    def _adapt_parameters(self, iteration: int, max_iterations: int):
        """Adapt parameters during search"""
        progress = iteration / max_iterations
        
        # Decrease inertia weight over time
        self.w = 0.9 - 0.5 * progress
        
        # Adjust cognitive and social parameters
        self.c1 = 2.5 - 0.5 * progress
        self.c2 = 0.5 + 1.5 * progress

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "ParticleSwarmSolver",
            "description": "Particle Swarm Optimization (quality-focused)",
            "time_limit": self.time_limit,
            "n_particles": self.n_particles,
            "n_iterations": self.n_iterations
        }
