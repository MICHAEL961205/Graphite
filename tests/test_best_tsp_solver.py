import asyncio
import time
from graphite.solvers import NearestNeighbourSolver
from graphite.solvers.best_tsp_solver import BestHeuristicTSPSolver
from graphite.data.dataset_utils import load_default_dataset
from graphite.data.dataset_generator_v2 import MetricTSPV2Generator
from graphite.protocol import GraphV2Synapse
from graphite.utils.graph_utils import get_tour_distance


def test_best_heuristic_tsp_solver_valid_tour():
    class Mock: pass
    mock = Mock()
    load_default_dataset(mock)

    problem = MetricTSPV2Generator.generate_one_sample(2000, mock.loaded_datasets)

    solver = BestHeuristicTSPSolver(time_limit=10.0)
    start = time.time()
    tour = asyncio.run(solver.solve_problem(problem))
    elapsed = time.time() - start

    syn = GraphV2Synapse(problem=problem)
    syn.solution = tour
    dist = get_tour_distance(syn)

    assert isinstance(tour, list) and len(tour) == problem.n_nodes + 1
    assert dist != float('inf')

    # Smoke-compare to NN (no assertion on quality to keep test stable)
    nn = NearestNeighbourSolver()
    tour_nn = asyncio.run(nn.solve_problem(problem))
    syn.solution = tour_nn
    _ = get_tour_distance(syn)


