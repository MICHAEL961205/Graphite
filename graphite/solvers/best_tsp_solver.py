from typing import List, Dict, Any
import numpy as np
import time

from graphite.protocol import GraphV2Problem
from graphite.solvers.common_utils import nearest_neighbor, two_opt_improve


class BestHeuristicTSPSolver:
    """
    Strong non-AI TSP solver:
    - Multistart initial solutions (NN from multiple seeds + random permutations)
    - 2-opt polish per start
    - Chained Lin–Kernighan refinement with double-bridge kicks
    Returns the best found tour within a time budget (adaptive by n).
    """

    def __init__(self, time_limit: float | None = None, max_starts: int | None = None):
        self.time_limit = time_limit
        self.max_starts = max_starts

    @staticmethod
    def get_solver_info() -> Dict[str, Any]:
        return {
            "name": "BestHeuristicTSPSolver",
            "type": "Heuristic",
            "notes": "Multistart + 2-opt + chained Lin–Kernighan",
        }

    async def solve_problem(self, problem: GraphV2Problem) -> List[int]:
        n = problem.n_nodes
        if not problem.edges or len(problem.edges) != n:
            return []

        dist = np.array(problem.edges, dtype=float)
        start_time = time.time()
        hard_limit = self._compute_time_limit(n) if self.time_limit is None else self.time_limit
        max_starts = self._compute_starts(n) if self.max_starts is None else self.max_starts

        best_tour: List[int] | None = None
        best_len = float("inf")

        # Diverse starts pool: NN from many seeds, random perms, Christofides+2opt if available
        seeds = list(range(min(64, n)))
        rng = np.random.default_rng(0)
        target_tries = min(100, max_starts)
        tries = 0

        def consider_tour(t):
            nonlocal best_tour, best_len
            L = self._tour_len(dist, t)
            if L < best_len:
                best_len, best_tour = L, t

        # optional christofides start
        if time.time() - start_time <= hard_limit:
            try:
                from graphite.solvers.christofides_two_opt_solver import ChristofidesTwoOptSolver  # type: ignore
                ch = ChristofidesTwoOptSolver()
                t = await ch.solve_problem(problem)
                if t and len(t) == n + 1:
                    consider_tour(t)
                    tries += 1
            except Exception:
                pass

        # Iterate through starts until time or tries exhausted
        si = 0
        while time.time() - start_time <= hard_limit and tries < target_tries:
            # Alternate between NN seed and random perm
            if si < len(seeds):
                s = seeds[si]
                si += 1
                base = nearest_neighbor(dist, start=s, start_time=start_time, hard_limit=hard_limit)
                base = self._ensure_permutation(base, n)
            else:
                perm = np.arange(n)
                rng.shuffle(perm)
                base = perm.tolist()

            tour = two_opt_improve(base + [base[0]], dist, start_time=start_time, hard_limit=hard_limit, max_iterations=100)
            tour = await self._lk_chain(problem, tour, start_time, hard_limit)
            consider_tour(tour)
            tries += 1

        if best_tour is None:
            base = nearest_neighbor(dist, start=0)
            best_tour = two_opt_improve(base + [base[0]], dist, max_iterations=50)
        if best_tour[0] != best_tour[-1]:
            best_tour = best_tour + [best_tour[0]]
        return best_tour

    async def _lk_refine(self, problem: GraphV2Problem, tour: List[int], start_time: float, hard_limit: float) -> List[int]:
        try:
            from graphite.solvers.chained_lin_kernighan_solver import ChainedLinKernighanSolver  # type: ignore
        except Exception:
            return tour
        budget = max(1.0, hard_limit - (time.time() - start_time))
        # Use a modest budget per refinement
        lk = ChainedLinKernighanSolver(time_limit=budget)
        return await lk.solve_problem(problem)

    async def _lk_chain(self, problem: GraphV2Problem, tour: List[int], start_time: float, hard_limit: float) -> List[int]:
        # Chain multiple LK passes with kicks while time remains
        current = tour
        while time.time() - start_time <= hard_limit:
            refined = await self._lk_refine(problem, current, start_time, hard_limit)
            if self._tour_len(np.array(problem.edges, dtype=float), refined) >= self._tour_len(np.array(problem.edges, dtype=float), current) - 1e-9:
                # perform a double-bridge kick
                kicked = self._double_bridge_kick(refined[:-1])
                if kicked[0] != kicked[-1]:
                    kicked = kicked + [kicked[0]]
                current = kicked
            else:
                current = refined
        return current

    @staticmethod
    def _double_bridge_kick(tour_no_close: List[int]) -> List[int]:
        n = len(tour_no_close)
        if n < 8:
            return tour_no_close
        rng = np.random.default_rng()
        a = int(rng.integers(1, n // 4))
        b = int(rng.integers(n // 4, n // 2))
        c = int(rng.integers(n // 2, 3 * n // 4))
        d = int(rng.integers(3 * n // 4, n - 1))
        p1 = tour_no_close[0:a]
        p2 = tour_no_close[a:b]
        p3 = tour_no_close[b:c]
        p4 = tour_no_close[c:d]
        p5 = tour_no_close[d:]
        return p1 + p3 + p2 + p4 + p5

    @staticmethod
    def _tour_len(dist: np.ndarray, tour: List[int]) -> float:
        total = 0.0
        for i in range(len(tour) - 1):
            total += float(dist[tour[i], tour[i + 1]])
        return total

    @staticmethod
    def _ensure_permutation(base: List[int], n: int) -> List[int]:
        if len(base) < n:
            missing = [i for i in range(n) if i not in set(base)]
            base = base + missing
        return base[:n]

    @staticmethod
    def _compute_time_limit(n: int) -> float:
        # Adaptive time limit by scale
        # Cap by 200s as the global time budget
        if n < 500:
            return min(200.0, 10.0)
        if n < 2000:
            return min(200.0, 60.0)
        return 200.0

    @staticmethod
    def _compute_starts(n: int) -> int:
        if n < 500:
            return 16
        if n < 2000:
            return 12
        return 8


