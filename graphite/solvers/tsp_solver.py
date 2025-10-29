# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
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
import math

import bittensor as bt

def NEAREST_NEIGHBOUR_SOLVER(formatted_problem, future_id:int)->List[int]:
    distance_matrix = formatted_problem
    n = len(distance_matrix[0])
    visited = [False] * n
    route = []
    total_distance = 0

    current_node = 0
    route.append(current_node)
    visited[current_node] = True

    for node in range(n - 1):
        # Find the nearest unvisited neighbour
        nearest_distance = np.inf
        nearest_node = random.choice([i for i, is_visited in enumerate(visited) if not is_visited])# pre-set as random unvisited node
        for j in range(n):
            if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                nearest_distance = distance_matrix[current_node][j]
                nearest_node = j

        # Move to the nearest unvisited node
        route.append(nearest_node)
        visited[nearest_node] = True
        total_distance += nearest_distance
        current_node = nearest_node
    
    # Return to the starting node
    total_distance += distance_matrix[current_node][route[0]]
    route.append(route[0])
    return route

class TSPSOLVER(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int)->List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])

        current = NEAREST_NEIGHBOUR_SOLVER(formatted_problem, future_id)

        # with open("input.txt", "w") as f:
        #     f.write(str(n) + "\n")
        #     f.write(" ".join([" ".join([str(x) for x in dst]) for dst in distance_matrix]) + "\n")
        #     f.write(" ".join(map(str, current)) + "\n")
        
        # exit(0)

        origin_len = sum(distance_matrix[current[i]][current[i + 1]] for i in range(n-1))


        best = current[:]
        best_len = origin_len
        origin = current[:-1]
        
        start_points = list(map(lambda x: x + 5, random.sample(range(n - 10), 1)))
        for point in start_points:

            now_node = origin[point:] + origin[:point]
            ST, ED = -1, -1
            for st in range(2, n - 3):
                for ed in range(st + 1, n - 2):
                    new_len = origin_len - distance_matrix[now_node[st - 1]][now_node[st]] - distance_matrix[now_node[ed]][now_node[ed + 1]] + distance_matrix[now_node[st - 1]][now_node[ed]] + distance_matrix[now_node[st]][now_node[ed + 1]]
                    if new_len < best_len:
                        ST, ED = st, ed
                        best_len = new_len
            
            if ST == -1 or ED == -1:
                continue
            new_node = now_node[:ST] + now_node[ED:ST - 1:-1] + now_node[ED+1:]
            best = new_node + [now_node[0]]
        return best

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges
        
if __name__=="__main__":
    # # runs the solver on a test MetricTSP
    # n_nodes = 100
    # test_problem = GraphV1Problem(n_nodes=n_nodes)
    # solver = NearestNeighbourSolver(problem_types=[test_problem])
    # start_time = time.time()
    # route = asyncio.run(solver.solve_problem(test_problem))
    # print(f"{solver.__class__.__name__} Solution: {route}")
    # print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")


    ## Test case for GraphV2Problem
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
      
    n_nodes = random.randint(2000, 5000)
    # randomly select n_nodes indexes from the selected graph
    selected_node_idxs = random.sample(range(26000000), n_nodes)
    test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB")
    if isinstance(test_problem, GraphV2Problem):
        test_problem.edges = recreate_edges(test_problem)
    print("Problem", test_problem)
    solver = TSPSOLVER(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")

    solver = TSPSOLVER(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
