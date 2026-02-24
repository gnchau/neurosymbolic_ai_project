import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import DimReductionConfig, EMBEDDINGS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DimensionalityReducer:
    def __init__(self, cfg: DimReductionConfig = DimReductionConfig()):
        self.cfg = cfg
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit_transform(
        self,
        train_embs: np.ndarray,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        d_target = target_dim or self.cfg.default_dim
        method = self.cfg.method
    
        logger.info(
            f"Fitting {method.upper()} | {train_embs.shape[1]}d -> {d_target}d "
            f"on {train_embs.shape[0]} samples"
        )
    
        train_scaled = self.scaler.fit_transform(train_embs)
    
        if method == "pca":
            self.model = PCA(n_components=d_target, random_state=42)
            reduced = self.model.fit_transform(train_scaled)
            explained_var = self.model.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_var:.4f} ({d_target} components)")
    
        elif method == "umap":
            self.model = umap.UMAP(
                n_components=d_target,
                n_neighbors=self.cfg.umap_n_neighbors,
                min_dist=self.cfg.umap_min_dist,
                metric=self.cfg.umap_metric,
                random_state=42,
            )
            reduced = self.model.fit_transform(train_scaled)
    
        elif method == "guide":
            from sklearn.decomposition import FastICA
            
            train_centered = train_scaled - np.mean(train_scaled, axis=0)
            
            M, T = train_centered.shape
            
            U, S, Vt = np.linalg.svd(train_centered, full_matrices=False)
            
            Uc = U[:, :d_target]
            Vc = Vt[:d_target, :].T
            
            UVc = np.concatenate((Uc, Vc), axis=0) * np.sqrt((M + T) / 2)
            
            ica = FastICA(
                max_iter=10000, 
                tol=0.000001, 
                random_state=42,
                whiten="arbitrary-variance"
            )
            
            ica_result = ica.fit_transform(UVc) / np.sqrt((M + T) / 2)
            
            W_XL, W_LT_t = np.split(ica_result, [M])
            W_LT = W_LT_t.T
            
            self.model = {
                'W_LT': W_LT,
                'train_mean': np.mean(train_scaled, axis=0)
            }
            
            reduced = train_centered @ W_LT.T
            
            logger.info(f"GUIDE reduction complete ({d_target} components)")
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
        self._is_fitted = True
        return reduced
        
    def transform(self, embs: np.ndarray) -> np.ndarray:
        """Transform new embeddings using the fitted reducer."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() first.")

        scaled = self.scaler.transform(embs)

        if self.cfg.method == "umap":
            return self.model.transform(scaled)
        elif self.cfg.method == "guide":
            # Center using the same mean from training
            centered = scaled - self.model['train_mean']
            return centered @ self.model['W_LT'].T
        else:
            # PCA and others
            return self.model.transform(scaled)
        
    
def reduce_embeddings(
    train_embs: np.ndarray,
    test_embs: np.ndarray,
    cfg: DimReductionConfig = DimReductionConfig(),
    target_dim: Optional[int] = None,
    cache_prefix: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, DimensionalityReducer]:
    d = target_dim or cfg.default_dim

    if cache_prefix:
        train_cache = EMBEDDINGS_DIR / f"{cache_prefix}_{cfg.method}_d{d}_train.npy"
        test_cache = EMBEDDINGS_DIR / f"{cache_prefix}_{cfg.method}_d{d}_test.npy"
        if train_cache.exists() and test_cache.exists():
            logger.info(f"Loading cached reduced embeddings (d={d})")
            return np.load(train_cache), np.load(test_cache), None

    reducer = DimensionalityReducer(cfg)
    train_reduced = reducer.fit_transform(train_embs, target_dim=d)
    test_reduced = reducer.transform(test_embs)

    if cache_prefix:
        np.save(train_cache, train_reduced)
        np.save(test_cache, test_reduced)
        logger.info(f"Cached reduced embeddings to {train_cache.parent}")

    return train_reduced, test_reduced, reducer
