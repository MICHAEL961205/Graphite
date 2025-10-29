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
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import tempfile
import os
import subprocess
import asyncio

try:
    from pyconcorde import Concorde
    PYCONCORDE_AVAILABLE = True
except ImportError:
    PYCONCORDE_AVAILABLE = False

class ConcordeSolver(BaseSolver):
    """
    Concorde TSP Solver implementation for the Graphite-Subnet project.
    
    This solver uses the Concorde TSP solver, which is one of the most powerful
    exact and heuristic TSP solvers available. It can handle both symmetric and
    asymmetric TSP instances.
    """
    
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = None, 
                 concorde_path: str = None, use_pyconcorde: bool = False, 
                 time_limit: int = 300):
        """
        Initialize the Concorde solver.
        
        Args:
            problem_types: List of problem types this solver can handle
            concorde_path: Path to Concorde executable (if not using pyconcorde)
            use_pyconcorde: Whether to use pyconcorde wrapper (recommended)
            time_limit: Maximum time limit in seconds for Concorde execution (default: 300)
        """
        if problem_types is None:
            problem_types = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]
        
        super().__init__(problem_types=problem_types)
        self.concorde_path = concorde_path or "concorde"
        self.use_pyconcorde = use_pyconcorde and PYCONCORDE_AVAILABLE
        self.time_limit = time_limit
        
        if not self.use_pyconcorde and not self._check_concorde_available():
            print("Warning: Concorde not available. Will use nearest neighbor fallback.")

    def _check_concorde_available(self) -> bool:
        """Check if Concorde executable is available."""
        try:
            result = subprocess.run([self.concorde_path, "-h"], 
                                  capture_output=True, text=True, timeout=5)
            # Concorde returns 1 for help and outputs usage to stderr
            return result.returncode == 1 and "Usage:" in result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        """
        Solve TSP using Concorde.
        
        Args:
            formatted_problem: Distance matrix as numpy array
            future_id: Unique identifier for this solve request
            
        Returns:
            List of node indices representing the optimal tour
        """
        distance_matrix = np.array(formatted_problem)
        n = len(distance_matrix)
        
        if n < 2:
            return [0] if n == 1 else []
        
        if self.use_pyconcorde:
            return await self._solve_with_pyconcorde(distance_matrix)
        elif self._check_concorde_available():
            return await self._solve_with_executable(distance_matrix)
        else:
            # Use nearest neighbor fallback
            return self._nearest_neighbor_fallback(distance_matrix)

    async def _solve_with_pyconcorde(self, distance_matrix: np.ndarray) -> List[int]:
        """Solve using pyconcorde wrapper."""
        def solve_sync():
            # Convert to integer distances (Concorde works better with integers)
            int_matrix = (distance_matrix * 1000).astype(int)
            
            # Create Concorde instance
            concorde = Concorde()
            
            # Solve the TSP
            tour, _ = concorde.solve(int_matrix)
            return tour.tolist()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, solve_sync)

    async def _solve_with_executable(self, distance_matrix: np.ndarray) -> List[int]:
        """Solve using Concorde executable directly."""
        def solve_sync():
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tsp', delete=False) as tsp_file:
                self._write_tsp_file(tsp_file.name, distance_matrix)
                
                # Create output file
                output_file = tsp_file.name.replace('.tsp', '.sol')
                
                try:
                    # Try different Concorde strategies that don't require LP solver
                    strategies = [
                        # Strategy 1: Just fast cuts with no branching
                        [self.concorde_path, "-V", "-B", "-w", "-o", output_file, tsp_file.name],
                        # Strategy 2: Just subtours and trivial blossoms
                        [self.concorde_path, "-w", "-o", output_file, tsp_file.name],
                        # Strategy 3: Fast cuts only
                        [self.concorde_path, "-V", "-o", output_file, tsp_file.name],
                    ]
                    
                    success = False
                    for cmd in strategies:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.time_limit)
                        if result.returncode == 0:
                            success = True
                            break
                        else:
                            print(f"Concorde strategy failed: {result.stderr[:200]}...")
                    
                    if not success:
                        print("All Concorde strategies failed, using improved fallback")
                        return self._nearest_neighbor_fallback(distance_matrix)
                    
                    # Read solution
                    tour = self._read_solution_file(output_file)
                    return tour
                    
                finally:
                    # Clean up temporary files
                    try:
                        os.unlink(tsp_file.name)
                        if os.path.exists(output_file):
                            os.unlink(output_file)
                    except OSError:
                        pass
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, solve_sync)

    def _nearest_neighbor_fallback(self, distance_matrix: np.ndarray) -> List[int]:
        """Advanced TSP solver using multiple heuristics and local search."""
        n = len(distance_matrix)
        
        if n < 2:
            return [0] if n == 1 else []
        
        best_route = None
        best_cost = float('inf')
        
        # Strategy 1: Multiple starting points with nearest neighbor + 2-opt
        max_starts = min(20, n)  # Try more starting points
        for start in range(max_starts):
            route = self._nearest_neighbor_from_start(start, distance_matrix)
            route = self._two_opt_improve(route, distance_matrix)
            route = self._three_opt_improve(route, distance_matrix)
            cost = self._calculate_route_cost(route, distance_matrix)
            
            if cost < best_cost:
                best_cost = cost
                best_route = route.copy()
        
        # Strategy 2: Greedy insertion
        route = self._greedy_insertion(distance_matrix)
        route = self._two_opt_improve(route, distance_matrix)
        route = self._three_opt_improve(route, distance_matrix)
        cost = self._calculate_route_cost(route, distance_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_route = route.copy()
        
        # Strategy 3: Simulated annealing
        route = self._simulated_annealing(distance_matrix)
        cost = self._calculate_route_cost(route, distance_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_route = route.copy()
        
        return best_route if best_route is not None else list(range(n))
    
    def _nearest_neighbor_from_start(self, start: int, distance_matrix: np.ndarray) -> List[int]:
        """Nearest neighbor starting from a specific node."""
        n = len(distance_matrix)
        visited = [False] * n
        route = [start]
        visited[start] = True
        current = start
        
        for _ in range(n - 1):
            nearest = -1
            min_dist = float('inf')
            for j in range(n):
                if not visited[j] and distance_matrix[current][j] < min_dist:
                    min_dist = distance_matrix[current][j]
                    nearest = j
            route.append(nearest)
            visited[nearest] = True
            current = nearest
        
        return route
    
    def _greedy_insertion(self, distance_matrix: np.ndarray) -> List[int]:
        """Greedy insertion heuristic for TSP."""
        n = len(distance_matrix)
        if n < 3:
            return list(range(n))
        
        # Start with the two closest nodes
        min_dist = float('inf')
        start1, start2 = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i][j] < min_dist:
                    min_dist = distance_matrix[i][j]
                    start1, start2 = i, j
        
        route = [start1, start2]
        unvisited = set(range(n)) - {start1, start2}
        
        while unvisited:
            best_node = None
            best_position = -1
            best_cost_increase = float('inf')
            
            for node in unvisited:
                for pos in range(len(route) + 1):
                    # Calculate cost increase of inserting node at position pos
                    if pos == 0:
                        cost_increase = (distance_matrix[node][route[0]] + 
                                       distance_matrix[route[-1]][node] - 
                                       distance_matrix[route[-1]][route[0]])
                    elif pos == len(route):
                        cost_increase = (distance_matrix[route[-1]][node] + 
                                       distance_matrix[node][route[0]] - 
                                       distance_matrix[route[-1]][route[0]])
                    else:
                        cost_increase = (distance_matrix[route[pos-1]][node] + 
                                       distance_matrix[node][route[pos]] - 
                                       distance_matrix[route[pos-1]][route[pos]])
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_node = node
                        best_position = pos
            
            route.insert(best_position, best_node)
            unvisited.remove(best_node)
        
        return route
    
    def _simulated_annealing(self, distance_matrix: np.ndarray) -> List[int]:
        """Simulated annealing for TSP."""
        n = len(distance_matrix)
        if n < 3:
            return list(range(n))
        
        # Start with nearest neighbor solution
        route = self._nearest_neighbor_from_start(0, distance_matrix)
        current_cost = self._calculate_route_cost(route, distance_matrix)
        
        # Simulated annealing parameters
        initial_temp = 1000.0
        final_temp = 0.1
        cooling_rate = 0.95
        max_iterations = 1000
        
        temp = initial_temp
        best_route = route.copy()
        best_cost = current_cost
        
        for iteration in range(max_iterations):
            # Generate neighbor by 2-opt swap
            i, j = np.random.randint(0, n, 2)
            if i > j:
                i, j = j, i
            
            # Create new route by reversing segment
            new_route = route.copy()
            new_route[i:j+1] = new_route[i:j+1][::-1]
            
            new_cost = self._calculate_route_cost(new_route, distance_matrix)
            
            # Accept or reject the move
            if new_cost < current_cost or np.random.random() < np.exp(-(new_cost - current_cost) / temp):
                route = new_route
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_route = route.copy()
                    best_cost = current_cost
            
            # Cool down
            temp *= cooling_rate
            if temp < final_temp:
                break
        
        return best_route
    
    def _three_opt_improve(self, route: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Apply 3-opt improvements to the route."""
        n = len(route)
        if n < 6:  # 3-opt needs at least 6 nodes
            return route
        
        improved = True
        while improved:
            improved = False
            best_route = route.copy()
            best_cost = self._calculate_route_cost(route, distance_matrix)
            
            # Try all possible 3-opt moves
            for i in range(1, n - 4):
                for j in range(i + 2, n - 2):
                    for k in range(j + 2, n):
                        # Try different 3-opt reconnections
                        for reconnection in self._three_opt_reconnections(route, i, j, k):
                            cost = self._calculate_route_cost(reconnection, distance_matrix)
                            if cost < best_cost:
                                best_cost = cost
                                best_route = reconnection
                                improved = True
            
            if improved:
                route = best_route
        
        return route
    
    def _three_opt_reconnections(self, route: List[int], i: int, j: int, k: int) -> List[List[int]]:
        """Generate all possible 3-opt reconnections."""
        n = len(route)
        reconnections = []
        
        # Original: A-B-C-D-E-F-G-H
        # After 3-opt: A-B-F-E-D-C-G-H (reverse C-D-E-F)
        new_route = route.copy()
        new_route[i:j+1] = new_route[i:j+1][::-1]
        reconnections.append(new_route)
        
        # A-B-C-D-E-F-G-H -> A-B-E-F-C-D-G-H (reverse C-D and E-F)
        new_route = route.copy()
        new_route[i:j+1] = new_route[i:j+1][::-1]
        new_route[j+1:k+1] = new_route[j+1:k+1][::-1]
        reconnections.append(new_route)
        
        # A-B-C-D-E-F-G-H -> A-B-F-E-D-C-G-H (reverse D-E-F)
        new_route = route.copy()
        new_route[j+1:k+1] = new_route[j+1:k+1][::-1]
        reconnections.append(new_route)
        
        return reconnections
    
    def _calculate_route_cost(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate the total cost of a route."""
        if len(route) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(route) - 1):
            cost += distance_matrix[route[i]][route[i + 1]]
        
        # Add cost to return to start
        cost += distance_matrix[route[-1]][route[0]]
        return cost
    
    def _two_opt_improve(self, route: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Apply 2-opt improvements to the route."""
        n = len(route)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Calculate current cost
                    current_cost = (distance_matrix[route[i-1]][route[i]] + 
                                  distance_matrix[route[j]][route[j+1] if j+1 < n else 0])
                    
                    # Calculate cost after swap
                    new_cost = (distance_matrix[route[i-1]][route[j]] + 
                               distance_matrix[route[i]][route[j+1] if j+1 < n else 0])
                    
                    if new_cost < current_cost:
                        # Perform the swap
                        route[i:j+1] = route[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        
        return route

    def _write_tsp_file(self, filename: str, distance_matrix: np.ndarray):
        """Write TSP instance in TSPLIB format."""
        n = len(distance_matrix)
        
        with open(filename, 'w') as f:
            f.write(f"NAME: generated_tsp\n")
            f.write(f"TYPE: TSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write(f"EDGE_WEIGHT_SECTION\n")
            
            # Convert to integer distances
            int_matrix = (distance_matrix * 1000).astype(int)
            
            for i in range(n):
                f.write(" ".join(str(int_matrix[i][j]) for j in range(n)) + "\n")
            
            f.write("EOF\n")

    def _read_solution_file(self, filename: str) -> List[int]:
        """Read Concorde solution file."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Concorde solution format: first line is dimension, second line is tour
        if len(lines) >= 2:
            tour_line = lines[1].strip()
            tour = [int(x) for x in tour_line.split()]
            return tour
        
        return []

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        """Transform problem to distance matrix format."""
        return problem.edges

    def get_solver_info(self) -> dict:
        """Get information about this solver."""
        return {
            "name": "ConcordeSolver",
            "description": "Concorde TSP Solver - exact and heuristic TSP solver",
            "use_pyconcorde": self.use_pyconcorde,
            "concorde_available": self._check_concorde_available() if not self.use_pyconcorde else True,
            "pyconcorde_available": PYCONCORDE_AVAILABLE,
            "time_limit": self.time_limit
        }
