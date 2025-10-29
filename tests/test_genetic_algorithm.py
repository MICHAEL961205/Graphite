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
from graphite.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver
from graphite.data.dataset_generator_v2 import MetricTSPV2Generator
from graphite.data.dataset_utils import load_default_dataset
from graphite.protocol import GraphV2Problem, GraphV2Synapse
from graphite.utils.graph_utils import get_tour_distance
import pandas as pd
import tqdm
import time
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

ROOT_DIR = "tests"
SAVE_DIR = "evaluation_results"
N_PROBLEMS = 6  # Smaller number for genetic algorithm due to complexity

def can_show_plot():
    # Check if running in a headless environment
    if os.name == 'posix':
        display = os.getenv('DISPLAY')
        if not display:
            return False

    # Check if the backend is suitable for interactive plotting
    backend = matplotlib.get_backend()
    if backend in ['agg', 'cairo', 'svg', 'pdf', 'ps']:
        return False

    return True

def compare_problems(solvers: List, problems: List[GraphV2Problem], loaded_datasets: dict):
    problem_types = set([problem.problem_type for problem in problems])
    mock_synapses = [GraphV2Synapse(problem=problem) for problem in problems]
    run_times_dict = {solver.__class__.__name__: [] for solver in solvers}
    scores_dict = {solver.__class__.__name__: [] for solver in solvers}
    
    for i, solver in enumerate(solvers):
        run_times = []
        scores = []
        print(f"Running Solver {i+1} - {solver.__class__.__name__}")
        if hasattr(solver, 'get_solver_info'):
            solver_info = solver.get_solver_info()
            print(f"Solver info: {solver_info}")
        
        for mock_synapse in tqdm.tqdm(mock_synapses, desc=f"{solver.__class__.__name__} solving {problem_types}"):
            # generate the edges adhoc
            MetricTSPV2Generator.recreate_edges(problem = mock_synapse.problem, loaded_datasets=loaded_datasets)
            start_time = time.perf_counter()
            
            try:
                # Run with timeout
                mock_synapse.solution = asyncio.run(asyncio.wait_for(
                    solver.solve_problem(mock_synapse.problem), 
                    timeout=100  # 100 second timeout
                ))
                run_time = time.perf_counter() - start_time
                
                if mock_synapse.solution is None:
                    run_time = float('inf')
                    score = float('inf')
                else:
                    score = get_tour_distance(mock_synapse)
                    
            except asyncio.TimeoutError:
                print(f"⚠️  {solver.__class__.__name__} timed out after 100 seconds")
                run_time = float('inf')
                score = float('inf')
                mock_synapse.solution = None
            except Exception as e:
                print(f"❌ {solver.__class__.__name__} failed with error: {e}")
                run_time = float('inf')
                score = float('inf')
                mock_synapse.solution = None

            run_times.append(run_time)
            scores.append(score)
            
            # remove edges and nodes to reduce memory consumption
            mock_synapse.problem.edges = None
            mock_synapse.problem.nodes = None
            
        run_times_dict[solver.__class__.__name__] = run_times
        scores_dict[solver.__class__.__name__] = scores
        
        # Print summary for this solver
        valid_times = [t for t in run_times if t != float('inf')]
        valid_scores = [s for s in scores if s != float('inf')]
        
        if valid_times:
            print(f"✅ {solver.__class__.__name__} completed {len(valid_times)}/{len(run_times)} problems")
            print(f"   Average time: {np.mean(valid_times):.2f}s")
            print(f"   Average score: {np.mean(valid_scores):.2f}")
        else:
            print(f"❌ {solver.__class__.__name__} failed all problems")
            
    return run_times_dict, scores_dict

def main():
    if not os.path.exists(os.path.join(ROOT_DIR, SAVE_DIR)):
        os.makedirs(os.path.join(ROOT_DIR, SAVE_DIR))

    # create Mock object to store the dataset
    class Mock:
        pass

    mock = Mock()
    load_default_dataset(mock) # load dataset as an attribute to mock instance

    # Use MetricTSPGenerator to generate problems of various graph sizes
    metric_problems, metric_sizes = MetricTSPV2Generator.generate_n_samples_without_edges(N_PROBLEMS, mock.loaded_datasets)

    print(f"\n{'='*60}")
    print(f"TESTING GeneticAlgorithmSolver vs NearestNeighbourSolver")
    print(f"Number of problems: {N_PROBLEMS}")
    print(f"{'='*60}")
    
    # Test solvers
    test_solvers = [GeneticAlgorithmSolver(time_limit=100), NearestNeighbourSolver()]
    run_times_dict, scores_dict = compare_problems(test_solvers, metric_problems, mock.loaded_datasets)

    # Create DataFrames for run times and scores
    run_times_df = pd.DataFrame(run_times_dict)
    scores_df = pd.DataFrame(scores_dict)

    # Add the problem size classification
    run_times_df['problem_size'] = metric_sizes
    scores_df['problem_size'] = metric_sizes

    # Set the problem index
    run_times_df.index.name = 'problem_index'
    scores_df.index.name = 'problem_index'

    # Save the data
    run_times_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "genetic_algorithm_run_times.csv"))
    scores_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "genetic_algorithm_scores.csv"))

    # Compare performance
    ga_scores = scores_dict['GeneticAlgorithmSolver']
    nn_scores = scores_dict['NearestNeighbourSolver']
    
    valid_ga = [s for s in ga_scores if s != float('inf')]
    valid_nn = [s for s in nn_scores if s != float('inf')]
    
    if valid_ga and valid_nn:
        improvements = sum(1 for g, n in zip(ga_scores, nn_scores) 
                          if g != float('inf') and n != float('inf') and g < n)
        total_comparable = sum(1 for g, n in zip(ga_scores, nn_scores) 
                             if g != float('inf') and n != float('inf'))
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"GeneticAlgorithmSolver improvements: {improvements}/{total_comparable} problems")
        print(f"GeneticAlgorithmSolver average score: {np.mean(valid_ga):.2f}")
        print(f"NearestNeighbourSolver average score: {np.mean(valid_nn):.2f}")
        
        if improvements > 0:
            print(f"✅ GeneticAlgorithmSolver is better than NearestNeighbourSolver!")
        else:
            print(f"❌ GeneticAlgorithmSolver is not better than NearestNeighbourSolver")
    else:
        print(f"❌ Cannot compare - insufficient valid results")

if __name__=="__main__":
    main()
