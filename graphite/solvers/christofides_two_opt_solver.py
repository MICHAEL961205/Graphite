# The MIT License (MIT)

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.solvers.christofides_solver import ChristofidesSolver
from graphite.solvers.common_utils import two_opt_improve
import time

class ChristofidesTwoOptSolver(BaseSolver):
    """
    Wrapper: run Christofides to build a tour, then apply 2-opt within time limit.
    """

    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, time_limit: int = 100):
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        start_time = time.time()
        dist = formatted_problem
        n = len(dist)
        if n <= 2:
            return list(range(n)) + [0]

        base = ChristofidesSolver(time_limit=max(1, int(self.time_limit * 0.7)))
        tour = await base.solve(formatted_problem, future_id)
        if tour and len(tour) > 1 and tour[-1] == tour[0]:
            tour = tour[:-1]

        tour = two_opt_improve(solution=tour, dist=dist, start_time=start_time, hard_limit=self.time_limit, max_iterations=20)
        tour.append(tour[0])
        return tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        return {
            "name": "ChristofidesTwoOptSolver",
            "description": "Christofides construction followed by 2-opt improvement",
            "time_limit": self.time_limit
        }


