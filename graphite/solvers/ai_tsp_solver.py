from typing import List, Optional, Dict, Any
import os
import math
import numpy as np

from graphite.protocol import GraphV2Problem
from .common_utils import nearest_neighbor, two_opt_improve


class AttentionModelSolver:
    """
    AI-based TSP solver scaffold that can load a pretrained attention/pointer policy
    and decode a tour. Falls back to nearest-neighbor + 2-opt when AI deps/weights
    are unavailable.

    Usage (optional AI path):
      - Set environment variable `AI_TSP_CHECKPOINT` to a local path of a compatible
        pretrained model checkpoint (e.g. Kool et al. Attention Model for TSP).
      - Alternatively, pass `checkpoint_path` to the constructor.

    Known pretrained sources you can use:
      - Kool et al. (Attention, Learn to Solve Routing Problems):
        https://github.com/wouterkool/attention-learn-to-route
        (Pretrained TSP20/50/100 models available in releases/issues.)
      - POMO (Policy Optimization with Multiple Optima):
        https://github.com/yd-kwon/POMO
        (Includes pretrained TSP models and inference scripts.)
      - NeuroLKH (ML-augmented LKH):
        https://github.com/liangxinedu/NeuroLKH

    Notes:
      - This scaffold expects Metric TSP with symmetric distance matrix in `problem.edges`.
      - Many academic models expect 2D coordinates; when only an edge matrix is present,
        the solver will fall back unless you adapt the model to accept edge weights.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, decode_strategy: str = "greedy"):
        self.checkpoint_path = checkpoint_path or os.getenv("AI_TSP_CHECKPOINT")
        self.decode_strategy = decode_strategy

        self._torch: Optional[Any] = None
        self._model: Optional[Any] = None

    @staticmethod
    def get_solver_info() -> Dict[str, Any]:
        return {
            "name": "AttentionModelSolver",
            "type": "AI",
            "requires": "PyTorch + compatible pretrained checkpoint (optional)",
            "env": "AI_TSP_CHECKPOINT for local .pt/.pth path",
        }

    async def solve_problem(self, problem: GraphV2Problem) -> List[int]:
        n = problem.n_nodes
        if not problem.edges or len(problem.edges) != n:
            # If edges are missing, we cannot run this solver; use a safe heuristic.
            return self._fallback(problem)

        if self._ensure_model_loaded() and self._model is not None:
            try:
                tour = self._infer_with_model(problem)
                if not self._is_valid_tour(tour, n):
                    return self._fallback(problem)
                # Ensure closed tour length n+1
                if tour[0] != tour[-1]:
                    tour = tour + [tour[0]]
                return tour
            except Exception:
                # Any runtime/model error → safe fallback
                return self._fallback(problem)
        else:
            return self._fallback(problem)

    def _fallback(self, problem: GraphV2Problem) -> List[int]:
        dist = np.array(problem.edges, dtype=float)
        initial = nearest_neighbor(dist, start=0)
        if len(initial) < problem.n_nodes:
            # Greedy fill any missing nodes
            missing = [i for i in range(problem.n_nodes) if i not in set(initial)]
            initial = initial + missing
        improved = two_opt_improve(initial + [initial[0]], dist, max_iterations=20)
        return improved

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None:
            return True
        # Lazy import torch to avoid mandatory dependency
        try:
            import torch  # type: ignore
        except Exception:
            return False

        self._torch = torch
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return False

        try:
            # This is a placeholder loader; adapt to your model class as needed.
            # Example for a scripted/traceable model:
            self._model = torch.jit.load(self.checkpoint_path, map_location="cpu")
            self._model.eval()
            return True
        except Exception:
            # If not a TorchScript model, users should adapt loader to their repo/model.
            self._model = None
            return False

    def _infer_with_model(self, problem: GraphV2Problem) -> List[int]:
        assert self._torch is not None and self._model is not None
        n = problem.n_nodes
        dist = np.array(problem.edges, dtype=np.float32)

        # Many attention models expect normalized 2D coordinates. If not available,
        # we attempt an edge-based decoding by providing the distance matrix.
        # This requires a model trained to consume an adjacency/edge matrix.
        # Input shape and API must match the loaded model.
        x = self._torch.from_numpy(dist).unsqueeze(0)  # shape: [1, n, n]

        with self._torch.no_grad():
            out = self._model(x)  # expected to return a permutation or logits

        # Support a few common output shapes:
        if isinstance(out, (list, tuple)):
            out = out[0]

        if out.dim() == 2 and out.shape[0] == n and out.shape[1] == 1:
            order = out.squeeze(1).cpu().numpy().astype(int).tolist()
        elif out.dim() == 2 and out.shape[0] == 1 and out.shape[1] == n:
            order = out.squeeze(0).cpu().numpy().astype(int).tolist()
        elif out.dim() == 1 and out.shape[0] == n:
            order = out.cpu().numpy().astype(int).tolist()
        else:
            # If model returns logits over next-city, do greedy argmax decoding
            if out.dim() == 3 and out.shape[-1] == n:
                # shape [1, n, n]: step-by-step logits
                logits = out.squeeze(0).cpu().numpy()
                order = self._greedy_decode_from_logits(logits)
            else:
                # Unknown format → fallback
                raise RuntimeError("Unsupported model output shape for tour decoding")

        # Ensure it's a permutation of all nodes
        if not self._is_valid_tour(order, n):
            raise RuntimeError("Model produced invalid tour")
        if order[0] != order[-1]:
            order = order + [order[0]]
        return order

    @staticmethod
    def _greedy_decode_from_logits(step_logits: np.ndarray) -> List[int]:
        n = step_logits.shape[0]
        visited = set()
        tour = []
        current = 0
        for _ in range(n):
            tour.append(current)
            visited.add(current)
            logits = step_logits[len(tour)-1]
            # mask visited
            masked = logits.copy()
            masked[list(visited)] = -1e9
            next_city = int(masked.argmax())
            if next_city in visited:
                # pick the best unvisited if argmax points to visited
                candidates = [(masked[j], j) for j in range(n) if j not in visited]
                if not candidates:
                    break
                next_city = max(candidates)[1]
            current = next_city
        # Append remaining if early stop
        if len(tour) < n:
            tour += [i for i in range(n) if i not in set(tour)]
        return tour

    @staticmethod
    def _is_valid_tour(tour: List[int], n: int) -> bool:
        if not tour:
            return False
        base = tour[:-1] if len(tour) == n + 1 and tour[0] == tour[-1] else tour
        return len(base) == n and set(base) == set(range(n)) and all(0 <= v < n for v in base)


