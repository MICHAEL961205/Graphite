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

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver import NearestNeighbourSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.utils.graph_utils import timeout
import numpy as np
import time
import asyncio
import random

import bittensor as bt

class ThreeOptSolver(BaseSolver):
    """
    3-opt TSP Solver implementation.
    
    This solver uses the 3-opt local search algorithm to improve TSP tours.
    It starts with a nearest neighbor solution and iteratively improves it
    by reconnecting three segments of the tour in different ways.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 time_limit: int = 100):
        """
        Initialize the 3-opt solver.
        
        Args:
            problem_types: List of problem types this solver can handle
            time_limit: Maximum time limit in seconds (default: 100)
        """
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        
        super().__init__(problem_types=problem_types)
        self.time_limit = time_limit

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        """
        Solve TSP using 3-opt algorithm.
        
        Args:
            formatted_problem: Distance matrix
            future_id: Future ID for tracking
            
        Returns:
            List of node indices representing the tour
        """
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        
        # Start with nearest neighbor solution
        nn_solver = NearestNeighbourSolver()
        current_tour = await nn_solver.solve(formatted_problem, future_id)
        
        if current_tour is None:
            return None
            
        # Remove the duplicate start node at the end
        if current_tour[-1] == current_tour[0]:
            current_tour = current_tour[:-1]
        
        start_time = time.time()
        improved = True
        
        while improved and (time.time() - start_time) < self.time_limit:
            if self.future_tracker.get(future_id):
                return None
                
            improved = False
            best_distance = self._calculate_tour_distance(current_tour, distance_matrix)
            
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        if (time.time() - start_time) >= self.time_limit:
                            break
                            
                        # Try all 3-opt reconnections
                        new_tours = self._get_three_opt_tours(current_tour, i, j, k)
                        
                        for new_tour in new_tours:
                            new_distance = self._calculate_tour_distance(new_tour, distance_matrix)
                            
                            if new_distance < best_distance:
                                current_tour = new_tour
                                best_distance = new_distance
                                improved = True
                                break
                                
                        if improved:
                            break
                            
                    if improved:
                        break
                        
        # Add the start node at the end to complete the cycle
        current_tour.append(current_tour[0])
        return current_tour

    def _get_three_opt_tours(self, tour, i, j, k):
        """Generate all possible 3-opt reconnections."""
        n = len(tour)
        tours = []
        
        # Original: A-B-C-D-E-F
        # After 3-opt cuts at i, j, k: A-B | C-D | E-F
        # We can reconnect in 7 different ways (excluding the original)
        
        # Tour 1: A-B-C-F-E-D
        tour1 = tour[:i] + tour[i:j] + tour[k:] + tour[j:k][::-1]
        tours.append(tour1)
        
        # Tour 2: A-B-E-D-C-F
        tour2 = tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:]
        tours.append(tour2)
        
        # Tour 3: A-B-E-F-C-D
        tour3 = tour[:i] + tour[j:k][::-1] + tour[k:] + tour[i:j]
        tours.append(tour3)
        
        # Tour 4: A-B-C-E-D-F
        tour4 = tour[:i] + tour[i:j] + tour[j:k][::-1] + tour[k:]
        tours.append(tour4)
        
        # Tour 5: A-B-D-C-E-F
        tour5 = tour[:i] + tour[i:j][::-1] + tour[j:k] + tour[k:]
        tours.append(tour5)
        
        # Tour 6: A-B-D-E-C-F
        tour6 = tour[:i] + tour[i:j][::-1] + tour[j:k][::-1] + tour[k:]
        tours.append(tour6)
        
        # Tour 7: A-B-F-E-D-C
        tour7 = tour[:i] + tour[k:] + tour[j:k][::-1] + tour[i:j][::-1]
        tours.append(tour7)
        
        return tours

    def _calculate_tour_distance(self, tour, distance_matrix):
        """Calculate the total distance of a tour."""
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        return total_distance

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

    def get_solver_info(self):
        """Return solver information."""
        return {
            "name": "ThreeOptSolver",
            "description": "3-opt local search algorithm for TSP",
            "time_limit": self.time_limit
        }

if __name__ == "__main__":
    # Test case
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
    loaded_datasets = {}
    with np.load('dataset/Asia_MSB.npz') as f:
        loaded_datasets["Asia_MSB"] = np.array(f['data'])
    
    def recreate_edges(problem: GraphV2Problem):
        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            return geom_edges(node_coords)
        elif problem.cost_function == "Euclidean2D":
            return euc_2d_edges(node_coords)
        elif problem.cost_function == "Manhatten2D":
            return man_2d_edges(node_coords)
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
      
    n_nodes = random.randint(20, 50)  # Smaller size for 3-opt due to complexity
    selected_node_idxs = random.sample(range(26000000), n_nodes)
    test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB")
    if isinstance(test_problem, GraphV2Problem):
        test_problem.edges = recreate_edges(test_problem)
    
    print("Problem", test_problem)
    solver = ThreeOptSolver(problem_types=[test_problem], time_limit=10)
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
