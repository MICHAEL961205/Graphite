from typing import List, Optional, Dict, Any
import os
import sys
import math
import numpy as np

# Ensure repo root is on sys.path when running this file directly
try:
    import graphite  # type: ignore
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from graphite.protocol import GraphV2Problem
try:
    from graphite.solvers.common_utils import nearest_neighbor, two_opt_improve
except Exception:
    # Fallback if executed within package context
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
        self._model_kind: Optional[str] = None  # 'torchscript' | 'kool'

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

        # For large instances, prefer strong heuristic (Chained Lin–Kernighan)
        if n >= 1000:
            try:
                tour = await self._lk_fallback(problem)
                if self._is_valid_tour(tour, n):
                    if tour[0] != tour[-1]:
                        tour = tour + [tour[0]]
                    return tour
            except Exception:
                pass

        # Medium range: clustering + heuristic stitching
        if n >= 300:
            try:
                tour = self._clustered_hybrid(problem)
                if self._is_valid_tour(tour, n):
                    if tour[0] != tour[-1]:
                        tour = tour + [tour[0]]
                    return tour
            except Exception:
                pass

        if self._ensure_model_loaded() and self._model is not None:
            try:
                print("Model loaded, inferring with model")
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
            print("Model not loaded, falling back to nearest neighbor")
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

        # Try TorchScript first
        try:
            self._model = torch.jit.load(self.checkpoint_path, map_location="cpu")
            self._model.eval()
            self._model_kind = 'torchscript'
            return True
        except Exception:
            self._model = None
            self._model_kind = None

        # Try Kool et al. repo format (folder with args.json + epoch-*.pt or .pt file)
        try:
            # Determine checkpoint dir/file
            ckpt_path = self.checkpoint_path
            ckpt_dir = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
            # Attempt to add repo root to sys.path if user cloned it
            kool_repo_root = os.getenv("AI_TSP_KOOL_REPO")
            if kool_repo_root and kool_repo_root not in sys.path:
                sys.path.insert(0, kool_repo_root)
            # If not provided, also try common default clone location
            for candidate in (kool_repo_root, "/root/attention-learn-to-route"):
                if candidate and os.path.isdir(candidate) and candidate not in sys.path:
                    sys.path.insert(0, candidate)
            # Import loader lazily
            from utils.functions import load_model as kool_load_model  # type: ignore
            model, args = kool_load_model(ckpt_dir)
            # Force greedy decoding
            try:
                model.set_decode_type("greedy", temp=1.0)  # type: ignore
            except Exception:
                pass
            self._model = model
            self._model_kind = 'kool'
            return True
        except Exception:
            self._model = None
            self._model_kind = None
            return False

    def _infer_with_model(self, problem: GraphV2Problem) -> List[int]:
        assert self._torch is not None and self._model is not None
        n = problem.n_nodes

        # Prefer coordinates if available (common for attention models)
        x = None
        if getattr(problem, "nodes", None):
            try:
                coords = np.array(problem.nodes, dtype=np.float32)
                if coords.ndim == 2 and coords.shape[1] >= 2:
                    min_xy = coords[:, :2].min(axis=0, keepdims=True)
                    max_xy = coords[:, :2].max(axis=0, keepdims=True)
                    denom = np.maximum(max_xy - min_xy, 1e-6)
                    norm_xy = (coords[:, :2] - min_xy) / denom
                    x = self._torch.from_numpy(norm_xy).unsqueeze(0)  # [1, n, 2]
            except Exception:
                x = None

        if x is None:
            # Fall back to distance-matrix input (requires compatible model)
            dist = np.array(problem.edges, dtype=np.float32)
            x = self._torch.from_numpy(dist).unsqueeze(0)  # [1, n, n]

        # Kool model path
        if self._model_kind == 'kool':
            # x must be [1, n, 2]
            if x.dim() != 3 or x.size(-1) != 2:
                raise RuntimeError("Kool model requires coordinates input of shape [1, n, 2]")
            model = self._model
            try:
                # Ensure eval/greedy
                model.eval()
                if hasattr(model, 'set_decode_type'):
                    model.set_decode_type("greedy", temp=1.0)  # type: ignore
                # sample_many returns (minpis, mincosts)
                pi, _ = model.sample_many(x, batch_rep=1, iter_rep=1)  # type: ignore
                order = pi[0].cpu().numpy().astype(int).tolist()
            except Exception as e:
                raise RuntimeError(f"Kool model inference failed: {e}")
            if not self._is_valid_tour(order, n):
                raise RuntimeError("Model produced invalid tour")
            if order[0] != order[-1]:
                order = order + [order[0]]
            return order

        # Generic TorchScript path
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

    async def _lk_fallback(self, problem: GraphV2Problem) -> List[int]:
        # Lazy import to avoid circular imports at module load
        try:
            from graphite.solvers.chained_lin_kernighan_solver import ChainedLinKernighanSolver  # type: ignore
        except Exception:
            return self._fallback(problem)
        solver = ChainedLinKernighanSolver()
        tour = await solver.solve_problem(problem)  # type: ignore
        return tour

    # -------------------- Large-n hybrid: clustering + NN + 2-opt --------------------
    def _clustered_hybrid(self, problem: GraphV2Problem) -> List[int]:
        """
        For large n, cluster nodes in coord space, solve each cluster with NN+2opt,
        order clusters by centroid tour, stitch, then global light 2-opt.
        """
        n = problem.n_nodes
        dist = np.array(problem.edges, dtype=float)

        # Require coordinates for meaningful clustering; otherwise fallback
        coords = None
        if getattr(problem, "nodes", None):
            try:
                arr = np.array(problem.nodes, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    coords = arr[:, :2]
            except Exception:
                coords = None
        if coords is None:
            return self._fallback(problem)

        # Normalize for stability
        min_xy = coords.min(axis=0, keepdims=True)
        max_xy = coords.max(axis=0, keepdims=True)
        denom = np.maximum(max_xy - min_xy, 1e-6)
        norm_xy = (coords - min_xy) / denom

        # Choose number of clusters ~ sqrt(n)
        k = max(1, int(np.sqrt(n)))
        clusters = self._kmeans_partition(norm_xy, k)

        # Solve intra-cluster tours
        cluster_tours: List[List[int]] = []
        centroids: List[np.ndarray] = []
        for idxs in clusters:
            if len(idxs) == 0:
                continue
            sub_idx = np.array(idxs, dtype=int)
            sub_dist = dist[np.ix_(sub_idx, sub_idx)]
            base = nearest_neighbor(sub_dist, start=0)
            if len(base) < len(sub_idx):
                missing = [i for i in range(len(sub_idx)) if i not in set(base)]
                base = base + missing
            cyc = two_opt_improve(base + [base[0]], sub_dist, max_iterations=(50 if len(sub_idx) <= 1000 else 10))
            # Map back to global indices, drop closing for stitching
            cyc_global = [int(sub_idx[i]) for i in cyc[:-1]]
            cluster_tours.append(cyc_global)
            centroids.append(norm_xy[sub_idx].mean(axis=0))

        if not cluster_tours:
            return self._fallback(problem)

        # Order clusters by centroid tour (NN on centroids)
        m = len(centroids)
        centroids_arr = np.stack(centroids, axis=0)
        # centroid distance matrix (Euclidean)
        dC = np.linalg.norm(centroids_arr[:, None, :] - centroids_arr[None, :, :], axis=-1)
        orderC = nearest_neighbor(dC, start=0)
        if len(orderC) < m:
            orderC += [i for i in range(m) if i not in set(orderC)]

        # Stitch clusters along centroid order using best boundary pair
        tour_linear: List[int] = []
        for t, cid in enumerate(orderC):
            seg = cluster_tours[cid]
            if t == 0:
                tour_linear = seg[:]
                continue
            prev_seg = tour_linear
            best = None
            # Evaluate connection between any node in prev_seg end section and any in seg
            # We consider connecting prev_end -> b and then append seg rotated to start at b
            for b in seg:
                db = dist[prev_seg[-1], b]
                if best is None or db < best[0]:
                    best = (db, b)
            _, best_b = best if best is not None else (0.0, seg[0])
            # rotate seg so it starts at best_b
            b_idx = seg.index(best_b)
            seg_rot = seg[b_idx:] + seg[:b_idx]
            tour_linear = prev_seg + seg_rot

        # Close and light 2-opt on full tour
        if tour_linear[0] != tour_linear[-1]:
            tour_linear = tour_linear + [tour_linear[0]]
        final = two_opt_improve(
            tour_linear, dist,
            max_iterations=(100 if n <= 1000 else 2)
        )
        return final

    @staticmethod
    def _kmeans_partition(xy: np.ndarray, k: int, iters: int = 10) -> List[List[int]]:
        n = xy.shape[0]
        if k >= n:
            return [[i] for i in range(n)]
        rng = np.random.default_rng(0)
        # kmeans++ init
        centers = [int(rng.integers(0, n))]
        for _ in range(1, k):
            d2 = np.min(np.sum((xy[:, None, :] - xy[np.array(centers)][None, :, :]) ** 2, axis=-1), axis=1)
            probs = d2 / (d2.sum() + 1e-12)
            centers.append(int(rng.choice(n, p=probs)))
        C = xy[np.array(centers)]
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            # assign
            d = np.sum((xy[:, None, :] - C[None, :, :]) ** 2, axis=-1)
            labels = d.argmin(axis=1)
            # update
            for j in range(k):
                pts = xy[labels == j]
                if len(pts) > 0:
                    C[j] = pts.mean(axis=0)
        clusters: List[List[int]] = [[] for _ in range(k)]
        for i, lab in enumerate(labels):
            clusters[int(lab)].append(int(i))
        # Remove empties
        clusters = [c for c in clusters if len(c) > 0]
        return clusters


