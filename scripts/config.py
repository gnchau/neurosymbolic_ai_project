import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

BASE_DIR = Path("/dartfs/rc/lab/B/BhattacharyaI/Results/Grant/convex")
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SELECTIONS_DIR = BASE_DIR / "selections"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

CACHE_DIR = BASE_DIR / "cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
TORCH_CACHE_DIR = CACHE_DIR / "torch"
SENTENCE_TRANSFORMERS_CACHE = CACHE_DIR / "sentence_transformers"

for d in [EMBEDDINGS_DIR, SELECTIONS_DIR, RESULTS_DIR, LOGS_DIR,
          HF_CACHE_DIR, TORCH_CACHE_DIR, SENTENCE_TRANSFORMERS_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_CACHE)
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
(CACHE_DIR / "tmp").mkdir(exist_ok=True)


MODEL_REGISTRY = {
    "qwen2-7b": {
        "hf_name": "Qwen/Qwen2-7B-Instruct",
        "short": "Qwen2-7B",
        "max_context_length": 32768,
    },
    "olmo-7b": {
        "hf_name": "allenai/OLMo-2-1124-7B-Instruct",
        "short": "OLMo-2-7B",
        "max_context_length": 4096,
    },
    "gemma2-9b": {
        "hf_name": "google/gemma-2-9b-it",
        "short": "Gemma2-9B",
        "max_context_length": 8192,
    },
    "gemma3-12b": {
        "hf_name": "google/gemma-3-12b-it",
        "short": "Gemma3-12B",
        "max_context_length": 32768,
    },
}

DATASET_REGISTRY = {
    "pubmedqa": {
        "task_type": "classification",
        "num_classes": 3,
        "labels": ["yes", "no", "maybe"],
        "domain": "medical",
    },
    "medqa": {
        "task_type": "multiple_choice",
        "num_classes": 4,
        "labels": ["A", "B", "C", "D"],
        "domain": "medical",
    },
    "medmcqa": {
        "task_type": "multiple_choice",
        "num_classes": 4,
        "labels": ["A", "B", "C", "D"],
        "domain": "medical",
    },
    "sst2": {
        "task_type": "classification",
        "num_classes": 2,
        "labels": ["negative", "positive"],
        "domain": "sentiment",
    },
    "agnews": {
        "task_type": "classification",
        "num_classes": 4,
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "domain": "topic",
    },
    "trec": {
        "task_type": "classification",
        "num_classes": 6,
        "labels": ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
        "domain": "question_type",
    },
}


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 128
    device: str = "cuda"
    max_seq_length: int = 512
    cache_dir: str = str(HF_CACHE_DIR)


@dataclass
class DimReductionConfig:
    method: str = "pca"
    target_dims: List[int] = field(default_factory=lambda: [10, 20, 50])
    default_dim: int = 20
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"


@dataclass
class SelectionConfig:
    k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    default_k: int = 5
    random_trials: int = 10
    ordering: str = "hull_distance"
    n_order_permutations: int = 5


@dataclass
class DatasetConfig:
    name: str = "pubmedqa"
    hf_path: str = "bigbio/pubmed_qa"
    hf_subset: str = "pubmed_qa_labeled_fold0_source"
    hf_path_alt: str = "qiaojin/PubMedQA"
    hf_subset_alt: str = "pqa_labeled"
    label_map: dict = field(default_factory=lambda: {"yes": 0, "no": 1, "maybe": 2})
    num_classes: int = 3
    cache_dir: str = str(HF_CACHE_DIR / "datasets")


@dataclass
class InferenceConfig:
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    model_key: str = "qwen2-7b"
    device: str = "cuda"
    max_new_tokens: int = 16
    temperature: float = 0.0
    batch_size: int = 1
    max_context_length: int = 32768  # default; overridden by from_registry
    cache_dir: str = str(HF_CACHE_DIR / "transformers")

    @classmethod
    def from_registry(cls, model_key: str) -> "InferenceConfig":
        entry = MODEL_REGISTRY[model_key]
        return cls(
            model_name=entry["hf_name"],
            model_key=model_key,
            max_context_length=entry.get("max_context_length", 4096),
        )


@dataclass
class EvalConfig:
    n_bootstrap_subsets: int = 10
    bootstrap_test_size: int = 200
    confidence_level: float = 0.95
    seed: int = 42


@dataclass
class ExperimentConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    dim_reduction: DimReductionConfig = field(default_factory=DimReductionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42