# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from graphite.solvers import *
from graphite.solvers.cheapest_insertion_two_opt_solver import CheapestInsertionTwoOptSolver
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

ALGONAME = "CheapestInsertionTwoOptSolver"
SCORES_CSV = f"{ALGONAME.replace('Solver','').lower()}_score.csv"  # matches: algorithm_name_score.csv
TIMES_CSV = f"{ALGONAME.replace('Solver','').lower()}_run_times.csv"


def compare_problems(solvers: List, problems: List[GraphV2Problem], loaded_datasets: dict):
    mock_synapses = [GraphV2Synapse(problem=problem) for problem in problems]
    run_times_dict = {solver.__class__.__name__: [] for solver in solvers}
    scores_dict = {solver.__class__.__name__: [] for solver in solvers}

    for solver in solvers:
        run_times = []
        scores = []
        for mock_synapse in tqdm.tqdm(mock_synapses, desc=f"{solver.__class__.__name__}"):
            MetricTSPV2Generator.recreate_edges(problem=mock_synapse.problem, loaded_datasets=loaded_datasets)
            start_time = time.perf_counter()
            try:
                mock_synapse.solution = asyncio.run(asyncio.wait_for(
                    solver.solve_problem(mock_synapse.problem, timeout=100),
                    timeout=100
                ))
                run_time = time.perf_counter() - start_time
                if mock_synapse.solution is None:
                    run_time = float('inf')
                    score = float('inf')
                else:
                    score = get_tour_distance(mock_synapse)
            except Exception:
                run_time = float('inf')
                score = float('inf')
                mock_synapse.solution = None
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

    test_solvers = [CheapestInsertionTwoOptSolver(time_limit=100), NearestNeighbourSolver()]
    run_times_dict, scores_dict = compare_problems(test_solvers, metric_problems, mock.loaded_datasets)

    run_times_df = pd.DataFrame(run_times_dict)
    scores_df = pd.DataFrame(scores_dict)
    run_times_df['problem_size'] = metric_sizes
    scores_df['problem_size'] = metric_sizes
    run_times_df.index.name = 'problem_index'
    scores_df.index.name = 'problem_index'

    # Save with required naming schema
    scores_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, SCORES_CSV))
    run_times_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, TIMES_CSV))

    algo_scores = scores_dict['CheapestInsertionTwoOptSolver']
    nn_scores = scores_dict['NearestNeighbourSolver']
    improvements = sum(1 for a, n in zip(algo_scores, nn_scores) if a < n)
    comparable = sum(1 for a, n in zip(algo_scores, nn_scores) if np.isfinite(a) and np.isfinite(n))

    print(f"{ALGONAME} improvements: {improvements}/{comparable}")


if __name__ == "__main__":
    main()
