import asyncio
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of nodes")
    parser.add_argument("--time", type=float, default=100.0, help="Time limit (s)")
    args = parser.parse_args()

    class Mock: pass
    mock = Mock()
    load_default_dataset(mock)

    problem = MetricTSPV2Generator.generate_one_sample(args.n, mock.loaded_datasets)

    import asyncio, time as _t
    from graphite.solvers import (
        NearestNeighbourSolver,
        ChristofidesTwoOptSolver,
        ChainedLinKernighanSolver,
        LKHSolver,
        MemeticSolver,
    )

    solvers = [
        ("NearestNeighbourSolver", NearestNeighbourSolver()),
        ("ChristofidesTwoOptSolver", ChristofidesTwoOptSolver(time_limit=args.time)),
        ("ChainedLinKernighanSolver", ChainedLinKernighanSolver(time_limit=args.time)),
        ("LKHSolver", LKHSolver(time_limit=args.time)),
        ("MemeticSolver", MemeticSolver(time_limit=args.time)),
        ("BestHeuristicTSPSolver", BestHeuristicTSPSolver(time_limit=args.time)),
    ]

    syn = GraphV2Synapse(problem=problem)

    results = []
    for name, solver in solvers:
        t0 = _t.time()
        tour = asyncio.run(solver.solve_problem(problem))
        dt = _t.time() - t0
        syn.solution = tour
        dist = get_tour_distance(syn)
        results.append((name, dt, dist))

    # Print neatly
    for name, dt, dist in results:
        print(f"{name:<26} n={args.n} time={dt:.2f}s dist={dist}")


