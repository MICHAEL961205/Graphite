# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphite.solvers import *
from graphite.solvers.christofides_solver import ChristofidesSolver
from graphite.data.dataset_generator_v2 import MetricTSPV2Generator
from graphite.data.dataset_utils import load_default_dataset
from graphite.protocol import GraphV2Problem, GraphV2Synapse
from graphite.utils.graph_utils import get_tour_distance
import pandas as pd
import tqdm
import time
import asyncio
import numpy as np

ROOT_DIR = "tests"
SAVE_DIR = "evaluation_results"
N_PROBLEMS = 10

def compare_problems(solvers, problems, loaded_datasets):
    mock_synapses = [GraphV2Synapse(problem=problem) for problem in problems]
    run_times_dict = {solver.__class__.__name__: [] for solver in solvers}
    scores_dict = {solver.__class__.__name__: [] for solver in solvers}

    for solver in solvers:
        run_times = []
        scores = []
        for mock_synapse in tqdm.tqdm(mock_synapses, desc=solver.__class__.__name__):
            MetricTSPV2Generator.recreate_edges(problem=mock_synapse.problem, loaded_datasets=loaded_datasets)
            start_time = time.perf_counter()
            try:
                mock_synapse.solution = asyncio.run(asyncio.wait_for(
                    solver.solve_problem(mock_synapse.problem, timeout=100),
                    timeout=100
                ))
                run_time = time.perf_counter() - start_time
                score = get_tour_distance(mock_synapse) if mock_synapse.solution else float('inf')
                if mock_synapse.solution is None:
                    run_time = float('inf')
                    score = float('inf')
            except Exception:
                run_time = float('inf')
                score = float('inf')
            run_times.append(run_time)
            scores.append(score)
            mock_synapse.problem.edges = None
            mock_synapse.problem.nodes = None
        run_times_dict[solver.__class__.__name__] = run_times
        scores_dict[solver.__class__.__name__] = scores
    return run_times_dict, scores_dict

def main():
    os.makedirs(os.path.join(ROOT_DIR, SAVE_DIR), exist_ok=True)

    class Mock:
        pass
    mock = Mock()
    load_default_dataset(mock)

    metric_problems, metric_sizes = MetricTSPV2Generator.generate_n_samples_without_edges(N_PROBLEMS, mock.loaded_datasets)

    test_solvers = [ChristofidesSolver(time_limit=100), NearestNeighbourSolver()]
    run_times_dict, scores_dict = compare_problems(test_solvers, metric_problems, mock.loaded_datasets)

    run_times_df = pd.DataFrame(run_times_dict)
    scores_df = pd.DataFrame(scores_dict)
    run_times_df['problem_size'] = metric_sizes
    scores_df['problem_size'] = metric_sizes
    run_times_df.index.name = 'problem_index'
    scores_df.index.name = 'problem_index'

    run_times_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "christofides_run_times.csv"))
    scores_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "christofides_scores.csv"))

    algo_scores = scores_dict['ChristofidesSolver']
    nn_scores = scores_dict['NearestNeighbourSolver']
    improvements = sum(1 for a, n in zip(algo_scores, nn_scores) 
                      if a != float('inf') and n != float('inf') and a < n)
    comparable = sum(1 for a, n in zip(algo_scores, nn_scores) 
                    if a != float('inf') and n != float('inf'))

    print(f"\nChristofidesSolver improvements: {improvements}/{comparable}")
    if improvements > 0:
        print(f"✅ ChristofidesSolver is better than NearestNeighbourSolver!")
    else:
        print(f"❌ ChristofidesSolver is not better than NearestNeighbourSolver")

if __name__ == "__main__":
    main()

