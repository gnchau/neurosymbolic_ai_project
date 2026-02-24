import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def facility_location_score(
    all_embeddings: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    # Wei et al. (2015) and used in Su et al. (2022) for ICL selection
    selected_embs = all_embeddings[selected_indices]
    sims = cosine_similarity(all_embeddings, selected_embs)
    max_sims = sims.max(axis=1)
    return float(max_sims.mean())


def test_coverage_score(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    selected_embs = train_embeddings[selected_indices]
    sims = cosine_similarity(test_embeddings, selected_embs)
    max_sims = sims.max(axis=1)
    return float(max_sims.mean())


def label_entropy(
    labels: List[str],
    selected_indices: np.ndarray,
    all_labels: Optional[List[str]] = None,
) -> float:
    selected_labels = [labels[i] for i in selected_indices]
    unique_labels = all_labels or list(set(labels))
    n_classes = len(unique_labels)

    if n_classes <= 1:
        return 0.0

    counts = np.array([selected_labels.count(l) for l in unique_labels], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_classes)
    return float(entropy / max_entropy)


def hull_volume_of_selection(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
) -> Optional[float]:
    selected_embs = embeddings[selected_indices]
    n, d = selected_embs.shape

    if n <= d:
        return None

    try:
        hull = ConvexHull(selected_embs)
        return float(hull.volume)
    except QhullError:
        return None


def mmd_score(
    all_embeddings: np.ndarray,
    selected_indices: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> float:
    selected_embs = all_embeddings[selected_indices]
    n_all = len(all_embeddings)
    n_sel = len(selected_embs)

    if gamma is None:
        dists = cdist(all_embeddings[:min(500, n_all)],
                      all_embeddings[:min(500, n_all)])
        gamma = 1.0 / (np.median(dists[dists > 0]) ** 2 + 1e-10)

    def rbf_kernel(X, Y):
        dists_sq = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-gamma * dists_sq)

    if n_all > 2000:
        rng = np.random.default_rng(42)
        subsample_idx = rng.choice(n_all, size=2000, replace=False)
        all_sub = all_embeddings[subsample_idx]
    else:
        all_sub = all_embeddings

    K_xx = rbf_kernel(all_sub, all_sub)
    K_yy = rbf_kernel(selected_embs, selected_embs)
    K_xy = rbf_kernel(all_sub, selected_embs)

    n_x = len(all_sub)
    n_y = n_sel

    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)

    mmd_sq = (
        K_xx.sum() / (n_x * (n_x - 1))
        + K_yy.sum() / (n_y * (n_y - 1)) if n_y > 1 else 0
    ) - 2 * K_xy.mean()

    return float(max(0, mmd_sq) ** 0.5)


def mean_test_nearest_sim(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    selected_embs = train_embeddings[selected_indices]
    sims = cosine_similarity(test_embeddings, selected_embs)
    return float(sims.max(axis=1).mean())


def compute_all_selection_metrics(
    train_embeddings_full: np.ndarray,
    test_embeddings_full: np.ndarray,
    selected_indices: np.ndarray,
    train_labels: List[str],
    all_label_names: List[str],
    train_embeddings_reduced: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {}

    # Diversity metrics (on full embeddings)
    selected_embs = train_embeddings_full[selected_indices]
    n = len(selected_indices)

    if n > 1:
        sim_matrix = cosine_similarity(selected_embs)
        triu_idx = np.triu_indices(n, k=1)
        metrics["mean_pairwise_sim"] = float(sim_matrix[triu_idx].mean())
    else:
        metrics["mean_pairwise_sim"] = 0.0

    gram = selected_embs @ selected_embs.T
    sign, logdet = np.linalg.slogdet(gram)
    metrics["gram_logdet"] = float(logdet) if sign > 0 else float("-inf")

    metrics["facility_location"] = facility_location_score(
        train_embeddings_full, selected_indices
    )
    metrics["test_coverage"] = test_coverage_score(
        train_embeddings_full, test_embeddings_full, selected_indices
    )
    metrics["mean_test_nearest_sim"] = mean_test_nearest_sim(
        train_embeddings_full, test_embeddings_full, selected_indices
    )

    metrics["mmd"] = mmd_score(train_embeddings_full, selected_indices)

    metrics["label_entropy"] = label_entropy(
        train_labels, selected_indices, all_labels=all_label_names
    )

    # Hull volume (only in reduced space)
    if train_embeddings_reduced is not None:
        vol = hull_volume_of_selection(train_embeddings_reduced, selected_indices)
        metrics["hull_volume"] = vol if vol is not None else float("nan")
    else:
        metrics["hull_volume"] = float("nan")

    metrics["n_selected"] = n
    return metrics