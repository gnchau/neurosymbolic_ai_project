import logging
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig, EMBEDDINGS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, cfg: EmbeddingConfig = EmbeddingConfig()):
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {cfg.model_name} on {self.device}")
        self.model = SentenceTransformer(
            cfg.model_name,
            device=self.device,
            cache_folder=cfg.cache_dir,
        )
        self.model.max_seq_length = cfg.max_seq_length

    def encode(
        self,
        texts: List[str],
        cache_name: Optional[str] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        if cache_name:
            cache_path = EMBEDDINGS_DIR / f"{cache_name}.npy"
            if cache_path.exists():
                logger.info(f"Loading cached embeddings from {cache_path}")
                embs = np.load(cache_path)
                if embs.shape[0] == len(texts):
                    return embs
                else:
                    logger.warning(f"Cache size mismatch ({embs.shape[0]} vs {len(texts)}), re-encoding")

        logger.info(f"Encoding {len(texts)} texts (batch_size={self.cfg.batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device,
        )

        if cache_name:
            cache_path = EMBEDDINGS_DIR / f"{cache_name}.npy"
            np.save(cache_path, embeddings)
            logger.info(f"Saved embeddings to {cache_path} | shape: {embeddings.shape}")

        return embeddings


def embed_dataset(data, format_fn, cfg, dataset_name):
    model_short = cfg.model_name.split("/")[-1]
    embedder = Embedder(cfg)
    embeddings = {}
    for split in ["train", "test"]:
        texts = [format_fn(ex) for ex in data[split]]
        cache_name = f"{dataset_name}_{split}_{model_short}"
        embeddings[split] = embedder.encode(texts, cache_name=cache_name)
        logger.info(f"{split} embeddings shape: {embeddings[split].shape}")
    return embeddings