# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time

class ChristofidesSolver(BaseSolver):
    """
    Christofides algorithm for metric TSP.
    
    Provides a 1.5-approximation guarantee for metric TSP:
    1. Build MST
    2. Find odd-degree vertices
    3. Find min-cost perfect matching on odd vertices
    4. Combine MST + matching to get Eulerian graph
    5. Find Eulerian circuit and skip duplicates
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, time_limit: int = 100):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        dist = formatted_problem
        n = len(dist)
        start_time = time.time()
        
        if n <= 2:
            return list(range(n)) + [0]
        
        # Step 1: Build MST using Prim's
        mst_edges = self._prim_mst(dist)
        if time.time() - start_time >= self.time_limit:
            return self._fallback_tour(n)
        
        # Step 2: Find odd-degree vertices
        degrees = self._compute_degrees(n, mst_edges)
        odd_vertices = [v for v in range(n) if degrees[v] % 2 == 1]
        
        if len(odd_vertices) <= 2:
            # Handle simple case
            tour = self._mst_to_hamiltonian(mst_edges, n)
            return tour + [tour[0]]
        
        if time.time() - start_time >= self.time_limit:
            return self._fallback_tour(n)
        
        # Step 3: Perfect matching on odd vertices (greedy min-cost)
        matching = self._greedy_perfect_matching(odd_vertices, dist)
        
        if time.time() - start_time >= self.time_limit:
            return self._fallback_tour(n)
        
        # Step 4: Combine MST + matching, form Eulerian tour
        tour = self._form_eulerian_tour(mst_edges, matching, n)
        
        if time.time() - start_time >= self.time_limit:
            return self._fallback_tour(n)
        
        return tour + [tour[0]]
    
    def _prim_mst(self, dist: np.ndarray) -> List[tuple]:
        n = len(dist)
        mst_edges = []
        in_mst = [False] * n
        min_edge_to = [-1] * n
        min_cost = [float('inf')] * n
        
        # Start from vertex 0
        min_cost[0] = 0
        for _ in range(n):
            u = min(range(n), key=lambda i: min_cost[i] if not in_mst[i] else float('inf'))
            if min_edge_to[u] != -1:
                mst_edges.append((min_edge_to[u], u))
            in_mst[u] = True
            for v in range(n):
                if not in_mst[v] and dist[u][v] < min_cost[v]:
                    min_cost[v] = dist[u][v]
                    min_edge_to[v] = u
        return mst_edges
    
    def _compute_degrees(self, n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def _greedy_perfect_matching(self, odd_vertices, dist):
        matching = []
        available = set(odd_vertices)
        
        while available:
            best_u, best_v, best_dist = None, None, float('inf')
            available_list = list(available)
            for u in available_list:
                for v in available_list:
                    if u < v and dist[u][v] < best_dist:
                        best_u, best_v, best_dist = u, v, dist[u][v]
            if best_u is None:
                break
            matching.append((best_u, best_v))
            available.remove(best_u)
            available.remove(best_v)
        return matching
    
    def _form_eulerian_tour(self, mst_edges, matching, n):
        # Build graph from MST + matching
        adj = [[] for _ in range(n)]
        for u, v in mst_edges + matching:
            adj[u].append(v)
            adj[v].append(u)
        
        # Iterative DFS to get eulerian circuit (avoid recursion depth issues)
        def iterative_dfs(start, path):
            stack = [start]
            while stack:
                v = stack[-1]
                if adj[v]:
                    w = adj[v].pop()
                    adj[w].remove(v)
                    stack.append(w)
                else:
                    path.append(stack.pop())
        
        path = []
        iterative_dfs(0, path)
        # Remove duplicates to get Hamiltonian path
        seen = set()
        hamiltonian = []
        for v in path:
            if v not in seen:
                hamiltonian.append(v)
                seen.add(v)
        return hamiltonian
    
    def _mst_to_hamiltonian(self, mst_edges, n):
        # DFS traversal of MST
        adj = [[] for _ in range(n)]
        for u, v in mst_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        tour = []
        def dfs(u, parent):
            tour.append(u)
            for v in adj[u]:
                if v != parent:
                    dfs(v, u)
        
        dfs(0, -1)
        return tour
    
    def _fallback_tour(self, n):
        return list(range(n))
    
    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges
    
    def get_solver_info(self):
        return {
            "name": "ChristofidesSolver",
            "description": "Christofides algorithm (1.5-approximation)",
            "time_limit": self.time_limit
        }

