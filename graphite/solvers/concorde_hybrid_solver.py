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

class ConcordeHybridSolver(BaseSolver):
    """
    Concorde Hybrid algorithm for TSP - quality-focused.
    
    Combines Concorde-style techniques with local search
    to find high-quality solutions within time limit.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100, max_iterations: int = 1000,
                 branch_factor: int = 2, depth_limit: int = 10):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.branch_factor = branch_factor
        self.depth_limit = depth_limit

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()
        hard_limit = self.time_limit  # Hard limit at exactly time_limit

        if n <= 2:
            return list(range(n)) + [0]

        # For small problems, use exact Concorde-style
        if n <= 15:
            return self._exact_concorde_style(distance_matrix, start_time, hard_limit)
        else:
            # For larger problems, use heuristic Concorde-style
            return self._heuristic_concorde_style(distance_matrix, start_time, hard_limit)

    def _exact_concorde_style(self, dist: np.ndarray, start_time: float, hard_limit: float) -> List[int]:
        """Exact Concorde-style algorithm for small problems"""
        n = len(dist)
        
        # Generate initial solution
        initial_solution = self._generate_initial_solution(n, dist, start_time)
        if initial_solution is None:
            return list(range(n)) + [0]
        
        best_solution = initial_solution
        best_cost = self._calculate_cost(best_solution, dist)
        
        # Apply Concorde-style branch and bound
        solution = self._concorde_branch_and_bound(best_solution, dist, start_time)
        if solution is not None:
            best_solution = solution
            best_cost = self._calculate_cost(best_solution, dist)
        
        return best_solution + [best_solution[0]]

    def _heuristic_concorde_style(self, dist: np.ndarray, start_time: float, hard_limit: float) -> List[int]:
        """Heuristic Concorde-style algorithm for larger problems"""
        n = len(dist)
        
        # Generate initial solution
        solution = self._generate_initial_solution(n, dist, start_time)
        if solution is None:
            return list(range(n)) + [0]
        
        # Apply Concorde-style improvements
        best_solution = self._concorde_improve(solution, dist, start_time)
        return best_solution + [best_solution[0]]

    def _generate_initial_solution(self, n: int, dist: np.ndarray, start_time: float) -> List[int]:
        """Generate initial solution using multiple strategies"""
        strategies = ['nearest_neighbor', 'christofides', 'greedy', 'random']
        
        for strategy in strategies:
            if (time.time() - start_time) >= hard_limit:
                break
                
            if strategy == 'nearest_neighbor':
                solution = self._nearest_neighbor(dist, start_time)
            elif strategy == 'christofides':
                solution = self._christofides_approx(dist, start_time)
            elif strategy == 'greedy':
                solution = self._greedy_construction(dist, start_time)
            elif strategy == 'random':
                solution = list(range(n))
                random.shuffle(solution)
            
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
            if (time.time() - start_time) >= hard_limit:
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

    def _concorde_branch_and_bound(self, initial_solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Concorde-style branch and bound"""
        n = len(initial_solution)
        best_solution = initial_solution[:]
        best_cost = self._calculate_cost(best_solution, dist)
        
        # Priority queue for nodes
        pq = []
        
        # Root node
        root_node = {
            'partial_tour': [],
            'remaining': set(range(n)),
            'lower_bound': self._calculate_lower_bound([], set(range(n)), dist),
            'cost': 0
        }
        pq.append((root_node['lower_bound'], root_node))
        
        nodes_explored = 0
        
        while pq and nodes_explored < self.max_iterations and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
                
            # Sort by lower bound
            pq.sort(key=lambda x: x[0])
            lower_bound, node = pq.pop(0)
            nodes_explored += 1
            
            # Prune if lower bound is worse than best solution
            if lower_bound >= best_cost:
                continue
            
            # If partial tour is complete, check if it's better
            if len(node['partial_tour']) == n:
                if node['cost'] < best_cost - 1e-12:
                    best_solution = node['partial_tour'][:]
                    best_cost = node['cost']
                continue
            
            # Branch: add each remaining city
            for city in node['remaining']:
                if (time.time() - start_time) >= hard_limit:
                    break
                    
                new_partial_tour = node['partial_tour'] + [city]
                new_remaining = node['remaining'] - {city}
                
                # Calculate cost of new partial tour
                new_cost = node['cost']
                if len(new_partial_tour) > 1:
                    new_cost += dist[new_partial_tour[-2]][new_partial_tour[-1]]
                
                # Calculate lower bound
                new_lower_bound = self._calculate_lower_bound(new_partial_tour, new_remaining, dist)
                
                # Only add if lower bound is promising
                if new_lower_bound < best_cost:
                    new_node = {
                        'partial_tour': new_partial_tour,
                        'remaining': new_remaining,
                        'lower_bound': new_lower_bound,
                        'cost': new_cost
                    }
                    pq.append((new_lower_bound, new_node))
        
        return best_solution

    def _concorde_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        """Concorde-style improvements"""
        n = len(solution)
        current_solution = solution[:]
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_iterations and (time.time() - start_time) < self.time_limit:
            improved = False
            iterations += 1
            
            # Try different improvement strategies
            strategies = ['2_opt', '3_opt', 'lin_kernighan', 'concorde_local']
            
            for strategy in strategies:
                if (time.time() - start_time) >= hard_limit:
                    break
                    
                if strategy == '2_opt':
                    if self._two_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == '3_opt':
                    if self._three_opt_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == 'lin_kernighan':
                    if self._lin_kernighan_improve(current_solution, dist, start_time):
                        improved = True
                        break
                elif strategy == 'concorde_local':
                    if self._concorde_local_improve(current_solution, dist, start_time):
                        improved = True
                        break
        
        return current_solution

    def _two_opt_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """2-opt improvement"""
        n = len(solution)
        improved = False
        
        for i in range(1, n - 2):
            if (time.time() - start_time) >= hard_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= hard_limit:
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
            if (time.time() - start_time) >= hard_limit:
                break
            for j in range(i + 1, n - 2):
                if (time.time() - start_time) >= hard_limit:
                    break
                for k in range(j + 1, n):
                    if (time.time() - start_time) >= hard_limit:
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

    def _lin_kernighan_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """Lin-Kernighan improvement"""
        n = len(solution)
        improved = False
        
        for start_city in range(min(10, n)):
            if (time.time() - start_time) >= hard_limit:
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
        for k in range(2, min(6, n // 2)):
            if (time.time() - start_time) >= hard_limit:
                break
                
            # Generate k-opt moves
            moves = self._generate_k_opt_moves(solution, start_pos, k, dist, start_time)
            
            for move in moves:
                if (time.time() - start_time) >= hard_limit:
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
            if (time.time() - start_time) >= hard_limit:
                break
            for j in range(i + 1, min(i + k + 1, n)):
                if (time.time() - start_time) >= hard_limit:
                    break
                # Try different reversal patterns
                new_solution = solution[:]
                new_solution[i:j] = reversed(new_solution[i:j])
                moves.append(new_solution)
        
        return moves

    def _concorde_local_improve(self, solution: List[int], dist: np.ndarray, start_time: float) -> bool:
        """Concorde-style local improvement"""
        n = len(solution)
        improved = False
        
        # Try different local improvement patterns
        for i in range(n - 3):
            if (time.time() - start_time) >= hard_limit:
                break
            for j in range(i + 2, n - 1):
                if (time.time() - start_time) >= hard_limit:
                    break
                for k in range(j + 2, n):
                    if (time.time() - start_time) >= hard_limit:
                        break
                    # Try different configurations
                    configs = [
                        solution[:i+1] + solution[j:k] + solution[i+1:j] + solution[k:],
                        solution[:i+1] + solution[j:k][::-1] + solution[i+1:j] + solution[k:],
                        solution[:i+1] + solution[j:k] + solution[i+1:j][::-1] + solution[k:]
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

    def _calculate_lower_bound(self, partial_tour: List[int], remaining: set, dist: np.ndarray) -> float:
        """Calculate lower bound for partial tour"""
        if not partial_tour:
            # Use minimum spanning tree as lower bound
            return self._mst_lower_bound(remaining, dist)
        
        n = len(dist)
        current_cost = 0
        
        # Cost of partial tour
        for i in range(len(partial_tour) - 1):
            current_cost += dist[partial_tour[i]][partial_tour[i + 1]]
        
        if not remaining:
            return current_cost
        
        # Add minimum cost to complete the tour
        # From last city in partial tour to first remaining city
        if remaining:
            min_outgoing = min(dist[partial_tour[-1]][city] for city in remaining)
            current_cost += min_outgoing
        
        # Add minimum spanning tree of remaining cities
        mst_cost = self._mst_lower_bound(remaining, dist)
        current_cost += mst_cost
        
        # Add minimum cost from remaining cities back to start
        if remaining:
            min_incoming = min(dist[city][partial_tour[0]] for city in remaining)
            current_cost += min_incoming
        
        return current_cost

    def _mst_lower_bound(self, cities: set, dist: np.ndarray) -> float:
        """Calculate minimum spanning tree lower bound"""
        if len(cities) <= 1:
            return 0.0
        
        cities_list = list(cities)
        n = len(cities_list)
        
        # Prim's algorithm for MST
        mst_cost = 0
        visited = {cities_list[0]}
        remaining = set(cities_list[1:])
        
        while remaining:
            min_edge = float('inf')
            min_city = None
            
            for visited_city in visited:
                for remaining_city in remaining:
                    edge_cost = dist[visited_city][remaining_city]
                    if edge_cost < min_edge:
                        min_edge = edge_cost
                        min_city = remaining_city
            
            if min_city is not None:
                mst_cost += min_edge
                visited.add(min_city)
                remaining.remove(min_city)
            else:
                break
        
        return mst_cost

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
            "name": "ConcordeHybridSolver",
            "description": "Concorde Hybrid algorithm",
            "time_limit": self.time_limit,
            "max_iterations": self.max_iterations,
            "branch_factor": self.branch_factor
        }
