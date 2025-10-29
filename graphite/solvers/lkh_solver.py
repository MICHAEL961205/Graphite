# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union, Tuple, Set
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time
import random

class LKHSolver(BaseSolver):
    """
    LKH (Lin-Kernighan-Helsgaun) algorithm for TSP - quality-focused.
    
    Advanced local search with multiple improvement strategies
    to find high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_trials: int = 1000, 
                 max_improvements: int = 50, k_opt_max: int = 5):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_trials = max_trials
        self.max_improvements = max_improvements
        self.k_opt_max = k_opt_max

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        # Generate initial solution
        solution = self._generate_initial_solution(n, distance_matrix, start_time)
        if solution is None:
            return list(range(n)) + [0]

        # Apply LKH improvements
        best_solution = self._lkh_improve(solution, distance_matrix, start_time)
        return best_solution + [best_solution[0]]

    def _generate_initial_solution(self, n: int, dist: np.ndarray, start_time: float) -> List[int]:
        """Generate initial solution using multiple strategies"""
        strategies = ['nearest_neighbor', 'random', 'greedy', 'christofides_approx']
        
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
            elif strategy == 'christofides_approx':
                solution = self._christofides_approx(dist, start_time)
            
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
            for city in remaining:
                if dist[current][city] < best_dist:
                    best_dist = dist[current][city]
                    nearest = city
            if nearest is None:
                break
            tour.append(nearest)
            remaining.remove(nearest)
            current = nearest
        return tour

    def _christofides_approx(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Christofides approximation algorithm"""
        n = len(dist)
        if n < 3:
            return list(range(n))
        
        # Find minimum spanning tree
        mst = self._find_mst(dist, start_time)
        if mst is None:
            return list(range(n))
        
        # Find odd-degree vertices
        odd_vertices = self._find_odd_vertices(mst, n)
        
        # Find minimum weight perfect matching
        matching = self._find_min_weight_matching(odd_vertices, dist, start_time)
        
        # Combine MST and matching
        multigraph = self._combine_mst_and_matching(mst, matching, n)
        
        # Find Eulerian tour
        eulerian_tour = self._find_eulerian_tour(multigraph, start_time)
        
        # Convert to Hamiltonian tour
        hamiltonian_tour = self._eulerian_to_hamiltonian(eulerian_tour)
        
        return hamiltonian_tour

    def _find_mst(self, dist: np.ndarray, start_time: float) -> List[Tuple[int, int]]:
        """Find minimum spanning tree using Prim's algorithm"""
        n = len(dist)
        mst = []
        visited = {0}
        remaining = set(range(1, n))
        
        while remaining and (time.time() - start_time) < self.time_limit:
            min_edge = None
            min_cost = float('inf')
            
            for u in visited:
                for v in remaining:
                    if dist[u][v] < min_cost:
                        min_cost = dist[u][v]
                        min_edge = (u, v)
            
            if min_edge is not None:
                mst.append(min_edge)
                visited.add(min_edge[1])
                remaining.remove(min_edge[1])
            else:
                break
        
        return mst

    def _find_odd_vertices(self, mst: List[Tuple[int, int]], n: int) -> List[int]:
        """Find vertices with odd degree in MST"""
        degree = [0] * n
        for u, v in mst:
            degree[u] += 1
            degree[v] += 1
        
        return [i for i in range(n) if degree[i] % 2 == 1]

    def _find_min_weight_matching(self, odd_vertices: List[int], dist: np.ndarray, start_time: float) -> List[Tuple[int, int]]:
        """Find minimum weight perfect matching (simplified)"""
        if len(odd_vertices) % 2 != 0:
            return []
        
        # Greedy matching
        remaining = odd_vertices[:]
        matching = []
        
        while len(remaining) >= 2 and (time.time() - start_time) < self.time_limit:
            min_cost = float('inf')
            min_pair = None
            
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    cost = dist[remaining[i]][remaining[j]]
                    if cost < min_cost:
                        min_cost = cost
                        min_pair = (remaining[i], remaining[j])
            
            if min_pair is not None:
                matching.append(min_pair)
                remaining.remove(min_pair[0])
                remaining.remove(min_pair[1])
            else:
                break
        
        return matching

    def _combine_mst_and_matching(self, mst: List[Tuple[int, int]], matching: List[Tuple[int, int]], n: int) -> List[List[int]]:
        """Combine MST and matching into multigraph"""
        multigraph = [[] for _ in range(n)]
        
        for u, v in mst:
            multigraph[u].append(v)
            multigraph[v].append(u)
        
        for u, v in matching:
            multigraph[u].append(v)
            multigraph[v].append(u)
        
        return multigraph

    def _find_eulerian_tour(self, multigraph: List[List[int]], start_time: float) -> List[int]:
        """Find Eulerian tour using Hierholzer's algorithm"""
        if not multigraph:
            return []
        
        # Find a vertex with odd degree, or start from 0
        start = 0
        for i, neighbors in enumerate(multigraph):
            if len(neighbors) % 2 == 1:
                start = i
                break
        
        tour = []
        stack = [start]
        
        while stack and (time.time() - start_time) < self.time_limit:
            current = stack[-1]
            if multigraph[current]:
                next_vertex = multigraph[current].pop()
                multigraph[next_vertex].remove(current)
                stack.append(next_vertex)
            else:
                tour.append(stack.pop())
        
        return tour[::-1]

    def _eulerian_to_hamiltonian(self, eulerian_tour: List[int]) -> List[int]:
        """Convert Eulerian tour to Hamiltonian tour"""
        visited = set()
        hamiltonian = []
        
        for vertex in eulerian_tour:
            if vertex not in visited:
                hamiltonian.append(vertex)
                visited.add(vertex)
        
        return hamiltonian

    def _lkh_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Apply LKH improvements"""
        n = len(solution)
        current_solution = solution[:]
        improved = True
        trials = 0
        
        while improved and trials < self.max_trials and (time.time() - start_time) < self.time_limit:
            improved = False
            trials += 1
            
            # Try different improvement strategies
            strategies = ['2_opt', '3_opt', '4_opt', '5_opt', 'or_opt', 'lin_kernighan']
            
            for strategy in strategies:
                if (time.time() - start_time) >= self.time_limit:
                    break
                    
                if strategy == '2_opt':
                    if self._two_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == '3_opt':
                    if self._three_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == '4_opt':
                    if self._four_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == '5_opt':
                    if self._five_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == 'or_opt':
                    if self._or_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == 'lin_kernighan':
                    if self._lin_kernighan_improve(current_solution, dist, start_time):
                        improved = True
                        break
        
        return current_solution

    def _two_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """2-opt improvement"""
        n = len(solution)
        improved = False
        
        for i in range(1, n - 2):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= self.time_limit:
                    break
                a, b = solution[i-1], solution[i]
                c, d = solution[j-1], solution[j]
                if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-12:
                    solution[i:j] = reversed(solution[i:j])
                    improved = True
                    break
            if improved:
                break
        return improved

    def _three_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """3-opt improvement"""
        n = len(solution)
        if n < 6:
            return self._two_opt_improve(solution, dist, start_time)
            
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
                        solution[:i] + solution[j:k] + solution[i:j] + solution[k:],
                        solution[:i] + solution[j:k][::-1] + solution[i:j] + solution[k:],
                        solution[:i] + solution[j:k] + solution[i:j][::-1] + solution[k:],
                        solution[:i] + solution[j:k][::-1] + solution[i:j][::-1] + solution[k:]
                    ]
                    
                    for config in configs:
                        if self._calculate_cost(config, dist) < self._calculate_cost(solution, dist) - 1e-12:
                            solution[:] = config
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return improved

    def _four_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """4-opt improvement"""
        n = len(solution)
        if n < 8:
            return self._three_opt_improve(solution, dist, start_time)
            
        improved = False
        for i in range(1, n - 6):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, n - 4):
                if (time.time() - start_time) >= self.time_limit:
                    break
                for k in range(j + 1, n - 2):
                    if (time.time() - start_time) >= self.time_limit:
                        break
                    for l in range(k + 1, n):
                        if (time.time() - start_time) >= self.time_limit:
                            break
                        # Try different 4-opt configurations
                        configs = [
                            solution[:i] + solution[j:k] + solution[l:] + solution[i:j] + solution[k:l],
                            solution[:i] + solution[j:k][::-1] + solution[l:] + solution[i:j] + solution[k:l],
                            solution[:i] + solution[j:k] + solution[l:][::-1] + solution[i:j] + solution[k:l],
                            solution[:i] + solution[j:k][::-1] + solution[l:][::-1] + solution[i:j] + solution[k:l]
                        ]
                        
                        for config in configs:
                            if self._calculate_cost(config, dist) < self._calculate_cost(solution, dist) - 1e-12:
                                solution[:] = config
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return improved

    def _five_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """5-opt improvement"""
        n = len(solution)
        if n < 10:
            return self._four_opt_improve(solution, dist, start_time)
            
        # Simplified 5-opt: try random 5-opt moves
        improved = False
        for _ in range(min(20, n)):
            if (time.time() - start_time) >= self.time_limit:
                break
            i, j, k, l, m = sorted(random.sample(range(n), 5))
            # Try different 5-opt configurations
            configs = [
                solution[:i] + solution[j:k] + solution[l:m] + solution[i:j] + solution[k:l] + solution[m:],
                solution[:i] + solution[j:k][::-1] + solution[l:m] + solution[i:j] + solution[k:l] + solution[m:],
                solution[:i] + solution[j:k] + solution[l:m][::-1] + solution[i:j] + solution[k:l] + solution[m:]
            ]
            
            for config in configs:
                if self._calculate_cost(config, dist) < self._calculate_cost(solution, dist) - 1e-12:
                    solution[:] = config
                    improved = True
                    break
            if improved:
                break
        return improved

    def _or_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """Or-opt improvement"""
        n = len(solution)
        improved = False
        
        for segment_length in range(1, min(4, n // 2)):
            if (time.time() - start_time) >= self.time_limit:
                break
            for start in range(n - segment_length + 1):
                if (time.time() - start_time) >= self.time_limit:
                    break
                segment = solution[start:start + segment_length]
                remaining = solution[:start] + solution[start + segment_length:]
                
                for insert_pos in range(len(remaining) + 1):
                    if (time.time() - start_time) >= self.time_limit:
                        break
                    new_solution = remaining[:insert_pos] + segment + remaining[insert_pos:]
                    if self._calculate_cost(new_solution, dist) < self._calculate_cost(solution, dist) - 1e-12:
                        solution[:] = new_solution
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    def _lin_kernighan_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """Lin-Kernighan improvement"""
        n = len(solution)
        improved = False
        
        for start_city in range(min(10, n)):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            # Try Lin-Kernighan moves starting from start_city
            current_solution = solution[:]
            if self._lin_kernighan_from_city(current_solution, start_city, dist, start_time):
                solution[:] = current_solution
                improved = True
                break
        
        return improved

    def _lin_kernighan_from_city(self, solution: List[int], start_city: int, dist: np.ndarray, start_time: float) -> bool:
        """Lin-Kernighan improvement starting from a specific city"""
        n = len(solution)
        improved = False
        
        # Find position of start_city
        start_pos = solution.index(start_city)
        
        # Try different Lin-Kernighan moves
        for k in range(2, min(self.k_opt_max + 1, n // 2)):
            if (time.time() - start_time) >= self.time_limit:
                break
                
            # Generate k-opt moves
            moves = self._generate_k_opt_moves(solution, start_pos, k, dist, start_time)
            
            for move in moves:
                if (time.time() - start_time) >= self.time_limit:
                    break
                if self._calculate_cost(move, dist) < self._calculate_cost(solution, dist) - 1e-12:
                    solution[:] = move
                    improved = True
                    break
            if improved:
                break
        
        return improved

    def _generate_k_opt_moves(self, solution: List[int], start_pos: int, k: int, dist: np.ndarray, start_time: float) -> List[List[int]]:
        """Generate k-opt moves"""
        n = len(solution)
        moves = []
        
        # Generate different k-opt configurations
        for i in range(start_pos, min(start_pos + 5, n - k)):
            if (time.time() - start_time) >= self.time_limit:
                break
            for j in range(i + 1, min(i + k + 1, n)):
                if (time.time() - start_time) >= self.time_limit:
                    break
                # Try different reversal patterns
                new_solution = solution[:]
                new_solution[i:j] = reversed(new_solution[i:j])
                moves.append(new_solution)
        
        return moves

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
            "name": "LKHSolver",
            "description": "LKH (Lin-Kernighan-Helsgaun) algorithm",
            "time_limit": self.time_limit,
            "max_trials": self.max_trials,
            "k_opt_max": self.k_opt_max
        }
