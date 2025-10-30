# The MIT License (MIT)

from typing import List, Union, Dict
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.solvers.common_utils import two_opt_improve
import numpy as np
import time

class ClarkeWrightSolver(BaseSolver):
    """
    Clarke–Wright Savings heuristic for symmetric metric TSP.
    Uses node 0 as depot and merges routes by descending savings.
    After construction, applies limited 2-opt improvement within time limit.
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

        # Initial routes: each node i has route [0, i, 0]
        routes: Dict[int, List[int]] = {i: [0, i, 0] for i in range(1, n)}

        # Compute savings s(i,j) = c(0,i) + c(0,j) - c(i,j)
        savings = []
        for i in range(1, n):
            for j in range(i + 1, n):
                s = dist[0][i] + dist[0][j] - dist[i][j]
                savings.append((s, i, j))
        savings.sort(reverse=True)

        # Track which route each node belongs to and its position (endpoints)
        node_to_route = {i: i for i in range(1, n)}

        def is_endpoint(route: List[int], node: int) -> bool:
            return (len(route) >= 2 and route[1] == node) or (len(route) >= 2 and route[-2] == node)

        # Merge routes by savings while respecting time and avoiding cycles
        for s, i, j in savings:
            if (time.time() - start_time) >= self.time_limit:
                break
            ri = node_to_route.get(i)
            rj = node_to_route.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]

            # Only merge at endpoints to maintain simple path
            if not is_endpoint(route_i, i) or not is_endpoint(route_j, j):
                continue

            # Determine orientation (i at end of route_i, j at start of route_j, etc.)
            # Strip depot at ends before concatenation
            if route_i[1] == i:  # i is at start side
                core_i = route_i[1:-1][::-1]
            else:  # i at end side
                core_i = route_i[1:-1]

            if route_j[-2] == j:  # j at end side
                core_j = route_j[1:-1]
            else:  # j at start side
                core_j = route_j[1:-1][::-1]

            # Merge core_i + core_j and wrap with depot
            merged_core = core_i + core_j
            # Check for duplicates to avoid cycles
            if len(set(merged_core)) != len(merged_core):
                continue

            merged = [0] + merged_core + [0]

            # Accept merge
            new_id = ri
            routes[new_id] = merged
            # Remove old rj
            del routes[rj]
            # Update node_to_route for nodes in route_j
            for node in core_j:
                node_to_route[node] = new_id
            for node in core_i:
                node_to_route[node] = new_id

        # Extract final tour from remaining route(s)
        if len(routes) == 1:
            tour = next(iter(routes.values()))
        else:
            # Fallback: concatenate remaining routes by nearest joins
            remaining = list(routes.values())
            current = remaining.pop(0)
            while remaining and (time.time() - start_time) < self.time_limit:
                # pick next route with cheapest connection
                best_k, best_cost, best_merge = -1, float('inf'), None
                for k, r in enumerate(remaining):
                    # try connecting ends: current end to r start, or r end to current start
                    opts = []
                    opts.append((current[:-1] + r[1:], dist[current[-2]][r[1]]))
                    opts.append((r[:-1] + current[1:], dist[r[-2]][current[1]]))
                    for seq, cost in opts:
                        if cost < best_cost:
                            best_cost = cost
                            best_k = k
                            best_merge = [0] + [x for x in seq if x != 0] + [0]
                if best_merge is None:
                    break
                current = best_merge
                remaining.pop(best_k)
            tour = current

        # Ensure tour validity length n+1
        core = [x for x in tour if x != 0]
        seen = set()
        dedup_core = []
        for x in core:
            if x not in seen:
                seen.add(x)
                dedup_core.append(x)
        # Add any missing nodes
        for node in range(1, n):
            if node not in seen:
                dedup_core.append(node)
        tour = [0] + dedup_core + [0]

        # 2-opt polish within remaining time
        tour = two_opt_improve(solution=tour[:-1], dist=dist, start_time=start_time, hard_limit=self.time_limit, max_iterations=10)
        tour.append(tour[0])
        return tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "ClarkeWrightSolver",
            "description": "Clarke–Wright Savings heuristic with 2-opt polish",
            "time_limit": self.time_limit
        }


