# The MIT License (MIT)

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import time

class MSTDoublingSolver(BaseSolver):
    """
    MST Doubling (2-approximation) heuristic for metric TSP:
    - Build MST
    - Perform preorder traversal (DFS) of MST to get permutation
    - Shortcut repeats, return to start
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

        # Prim's algorithm for MST
        in_mst = [False] * n
        key = [float('inf')] * n
        parent = [-1] * n
        key[0] = 0.0
        for _ in range(n):
            if (time.time() - start_time) >= self.time_limit:
                break
            u = -1
            best = float('inf')
            for v in range(n):
                if not in_mst[v] and key[v] < best:
                    best = key[v]
                    u = v
            if u == -1:
                break
            in_mst[u] = True
            for v in range(n):
                w = dist[u][v]
                if not in_mst[v] and w < key[v]:
                    key[v] = w
                    parent[v] = u

        # Build adjacency list for MST
        adj = [[] for _ in range(n)]
        for v in range(1, n):
            if parent[v] != -1:
                adj[parent[v]].append(v)
                adj[v].append(parent[v])

        # Preorder DFS traversal starting at 0
        order = []
        stack = [0]
        visited = [False] * n
        while stack and (time.time() - start_time) < self.time_limit:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            order.append(u)
            # push neighbors in reverse to get natural order
            for v in reversed(adj[u]):
                if not visited[v]:
                    stack.append(v)

        # Shortcut repeated visits
        seen = set()
        tour = []
        for v in order:
            if v not in seen:
                seen.add(v)
                tour.append(v)
        # Ensure all nodes included
        for v in range(n):
            if v not in seen:
                tour.append(v)
        tour.append(tour[0])
        return tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "MSTDoublingSolver",
            "description": "MST Doubling heuristic (2-approx) for metric TSP",
            "time_limit": self.time_limit
        }


