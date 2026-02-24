import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from config import SelectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    method: str
    indices: np.ndarray
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def select_convex_hull(embeddings: np.ndarray, k: int, return_all_vertices: bool = False) -> SelectionResult:
    n, d = embeddings.shape
    logger.info(f"Computing convex hull: {n} points in {d}d")

    try:
        hull = ConvexHull(embeddings)
    except QhullError as e:
        logger.warning(f"Convex hull failed ({e}), falling back to farthest-first traversal")
        return select_farthest_first(embeddings, k)

    vertex_indices = np.unique(hull.vertices)
    n_vertices = len(vertex_indices)
    logger.info(f"Convex hull has {n_vertices} vertices (requested k={k})")

    center = embeddings[vertex_indices].mean(axis=0)
    vertex_dists = np.linalg.norm(embeddings[vertex_indices] - center, axis=1)

    if return_all_vertices:
        order = np.argsort(-vertex_dists)
        selected = vertex_indices[order]
    elif k <= n_vertices:
        selected = _farthest_first_from_pool(embeddings[vertex_indices], k)
        selected = vertex_indices[selected]
    else:
        non_vertex_mask = np.ones(n, dtype=bool)
        non_vertex_mask[vertex_indices] = False
        non_vertex_indices = np.where(non_vertex_mask)[0]

        if len(non_vertex_indices) > 0:
            dists_to_hull = cdist(embeddings[non_vertex_indices], embeddings[vertex_indices]).min(axis=1)
            n_extra = k - n_vertices
            extra_order = np.argsort(dists_to_hull)[:n_extra]
            extra_indices = non_vertex_indices[extra_order]
            selected = np.concatenate([vertex_indices, extra_indices])
        else:
            selected = vertex_indices[:k]

    dists_from_center = np.linalg.norm(embeddings[selected] - center, axis=1)
    order = np.argsort(-dists_from_center)
    selected = selected[order][:k]

    return SelectionResult(
        method="convex_hull",
        indices=selected,
        metadata={
            "n_vertices": n_vertices,
            "hull_volume": hull.volume if hasattr(hull, "volume") else None,
            "center": center,
            "vertex_indices": vertex_indices,
        },
    )


def select_random(n_samples: int, k: int, n_trials: int = 10, seed: int = 42) -> List[SelectionResult]:
    rng = np.random.default_rng(seed)
    results = []
    for t in range(n_trials):
        indices = rng.choice(n_samples, size=k, replace=False)
        results.append(SelectionResult(method="random", indices=indices, metadata={"trial": t}))
    return results


def select_random_single(n_samples: int, k: int, seed: int = 42) -> SelectionResult:
    """Single random selection (for sweep use)."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, size=k, replace=False)
    return SelectionResult(method="random", indices=indices, metadata={"seed": seed})


def select_knn(train_embeddings: np.ndarray, query_embedding: np.ndarray, k: int) -> SelectionResult:
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    sims = cosine_similarity(query_embedding, train_embeddings)[0]
    top_k = np.argsort(-sims)[:k]
    return SelectionResult(method="knn", indices=top_k, metadata={"similarities": sims[top_k]})


def select_kmeans(embeddings: np.ndarray, k: int, seed: int = 42) -> SelectionResult:
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(embeddings)
    dists = cdist(kmeans.cluster_centers_, embeddings)
    selected = np.argmin(dists, axis=1)
    selected = np.unique(selected)
    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(len(embeddings)), selected)
        dists_remaining = cdist(kmeans.cluster_centers_, embeddings[remaining]).min(axis=0)
        extra = remaining[np.argsort(dists_remaining)[: k - len(selected)]]
        selected = np.concatenate([selected, extra])
    return SelectionResult(method="kmeans", indices=selected[:k], metadata={"inertia": kmeans.inertia_})


def _farthest_first_from_pool(embeddings: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    n = len(embeddings)
    rng = np.random.default_rng(seed)
    selected = [rng.integers(n)]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        last = embeddings[selected[-1]]
        dists = np.linalg.norm(embeddings - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists_masked = min_dists.copy()
        min_dists_masked[selected] = -1
        selected.append(np.argmax(min_dists_masked))

    return np.array(selected)


def select_farthest_first(embeddings: np.ndarray, k: int, seed: int = 42) -> SelectionResult:
    indices = _farthest_first_from_pool(embeddings, k, seed=seed)
    return SelectionResult(method="farthest_first", indices=indices, metadata={})


def select_dpp(embeddings: np.ndarray, k: int, seed: int = 42) -> SelectionResult:
    n = len(embeddings)
    L = embeddings @ embeddings.T
    rng = np.random.default_rng(seed)
    selected = []
    remaining = list(range(n))

    for _ in range(k):
        if not selected:
            probs = np.diag(L)[remaining]
            probs = probs / probs.sum()
            idx = rng.choice(remaining, p=probs)
        else:
            best_score = -np.inf
            best_idx = remaining[0]
            sel_arr = np.array(selected)
            L_sel = L[np.ix_(sel_arr, sel_arr)]
            current_logdet = np.linalg.slogdet(L_sel)[1]

            for candidate in remaining:
                new_sel = np.append(sel_arr, candidate)
                L_new = L[np.ix_(new_sel, new_sel)]
                try:
                    new_logdet = np.linalg.slogdet(L_new)[1]
                except np.linalg.LinAlgError:
                    continue
                if new_logdet > best_score:
                    best_score = new_logdet
                    best_idx = candidate
            idx = best_idx

        selected.append(idx)
        remaining.remove(idx)

    return SelectionResult(method="dpp", indices=np.array(selected), metadata={})


def select_facility_location(
    embeddings: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> SelectionResult:
    n = len(embeddings)
    logger.info(f"Facility location: selecting {k} from {n} points")

    if metric == "cosine":
        sim_matrix = cosine_similarity(embeddings)
    else:
        dists = cdist(embeddings, embeddings, metric=metric)
        sim_matrix = -dists

    selected = []
    # current_max_sims[i] = max similarity from point i to any selected point
    current_max_sims = np.full(n, -np.inf)
    remaining = set(range(n))

    for step in range(k):
        best_gain = -np.inf
        best_idx = -1

        for candidate in remaining:
            new_sims = np.maximum(current_max_sims, sim_matrix[candidate])
            gain = new_sims.mean() - current_max_sims.mean()

            if gain > best_gain:
                best_gain = gain
                best_idx = candidate

        selected.append(best_idx)
        remaining.discard(best_idx)
        current_max_sims = np.maximum(current_max_sims, sim_matrix[best_idx])

        if (step + 1) % 10 == 0:
            logger.info(
                f"  Facility location step {step + 1}/{k}: "
                f"coverage={current_max_sims.mean():.4f}"
            )

    final_coverage = current_max_sims.mean()
    logger.info(f"Facility location done: coverage={final_coverage:.4f}")

    return SelectionResult(
        method="facility_location",
        indices=np.array(selected),
        metadata={"final_coverage": float(final_coverage)},
    )



class BM25Selector:
    def __init__(self, corpus: List[str]):
        from rank_bm25 import BM25Okapi
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def select(self, query: str, k: int) -> SelectionResult:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k = np.argsort(-scores)[:k]
        return SelectionResult(method="bm25", indices=top_k, metadata={"scores": scores[top_k]})



def order_by_hull_distance(indices: np.ndarray, embeddings: np.ndarray, center: Optional[np.ndarray] = None) -> np.ndarray:
    if center is None:
        center = embeddings[indices].mean(axis=0)
    dists = np.linalg.norm(embeddings[indices] - center, axis=1)
    return indices[np.argsort(-dists)]


def average_over_orderings(indices: np.ndarray, n_permutations: int = 5, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.permutation(indices) for _ in range(n_permutations)]


def compute_diversity_metrics(embeddings: np.ndarray, indices: np.ndarray) -> Dict:
    selected_embs = embeddings[indices]
    n = len(indices)

    if n > 1:
        sim_matrix = cosine_similarity(selected_embs)
        triu_idx = np.triu_indices(n, k=1)
        mean_pairwise_sim = sim_matrix[triu_idx].mean()
    else:
        mean_pairwise_sim = 0.0

    gram = selected_embs @ selected_embs.T
    sign, logdet = np.linalg.slogdet(gram)
    gram_logdet = logdet if sign > 0 else -np.inf

    centroid = selected_embs.mean(axis=0)
    mean_dist_from_centroid = np.linalg.norm(selected_embs - centroid, axis=1).mean()

    return {
        "mean_pairwise_cosine_sim": float(mean_pairwise_sim),
        "gram_logdet": float(gram_logdet),
        "mean_dist_from_centroid": float(mean_dist_from_centroid),
        "n_selected": n,
    }