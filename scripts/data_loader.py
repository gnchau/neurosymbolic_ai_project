import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import DatasetConfig, BASE_DIR, HF_CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = str(HF_CACHE_DIR / "datasets")


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}_processed.json"


def _save_cache(name: str, data: dict):
    path = _cache_path(name)
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info(f"Cached to {path}")


def _load_cache(name: str):
    path = _cache_path(name)
    if path.exists():
        logger.info(f"Loading cached {name} from {path}")
        with open(path) as f:
            return json.load(f)
    return None


def _stratified_split(examples, test_size=0.2, seed=42):
    labels = [ex["label"] for ex in examples]
    train_idx, test_idx = train_test_split(
        range(len(examples)), test_size=test_size, stratify=labels, random_state=seed
    )
    return {
        "train": [examples[i] for i in train_idx],
        "test": [examples[i] for i in test_idx],
    }


def _log_stats(name, data):
    for split in ["train", "test"]:
        dist = {}
        for ex in data[split]:
            lbl = ex.get("answer", ex.get("label_str", "?"))
            dist[lbl] = dist.get(lbl, 0) + 1
        logger.info(f"{name} {split}: {len(data[split])} examples, distribution: {dist}")


def load_pubmedqa(cfg: DatasetConfig = DatasetConfig()) -> Dict[str, List[dict]]:
    cached = _load_cache("pubmedqa")
    if cached:
        return cached

    ds = None
    for path, subset in [
        (cfg.hf_path_alt, cfg.hf_subset_alt),
        (cfg.hf_path, cfg.hf_subset),
    ]:
        try:
            ds = load_dataset(path, subset, trust_remote_code=True, cache_dir=cfg.cache_dir)
            logger.info(f"Loaded from {path}/{subset}")
            break
        except Exception as e:
            logger.warning(f"Failed {path}/{subset}: {e}")

    if ds is None:
        raise RuntimeError("Could not load PubMedQA")

    split_name = list(ds.keys())[0]
    examples = []
    for i, row in enumerate(ds[split_name]):
        question = row.get("question", row.get("QUESTION", ""))
        context_raw = row.get("context", row.get("CONTEXTS", row.get("long_answer", "")))
        if isinstance(context_raw, dict):
            context_raw = context_raw.get("contexts", context_raw.get("sentences", ""))
        if isinstance(context_raw, list):
            flat = []
            for item in context_raw:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(str(item))
            context = " ".join(flat)
        else:
            context = str(context_raw)

        answer = str(row.get("final_decision", row.get("answer", ""))).lower().strip()
        if answer not in cfg.label_map:
            continue

        examples.append({
            "id": str(row.get("pubid", row.get("id", i))),
            "question": question,
            "context": context[:2000],
            "answer": answer,
            "label": cfg.label_map[answer],
        })

    data = _stratified_split(examples)
    _log_stats("pubmedqa", data)
    _save_cache("pubmedqa", data)
    return data


def load_medqa() -> Dict[str, List[dict]]:
    cached = _load_cache("medqa")
    if cached:
        return cached

    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", cache_dir=CACHE_DIR)

    def _process(split):
        examples = []
        for i, row in enumerate(split):
            question = row.get("sent1", row.get("question", ""))
            option_dict = {
                "A": row.get("ending0", row.get("opa", "")),
                "B": row.get("ending1", row.get("opb", "")),
                "C": row.get("ending2", row.get("opc", "")),
                "D": row.get("ending3", row.get("opd", "")),
            }
            label = row.get("label", None)
            if label is None or label not in [0, 1, 2, 3]:
                continue
            answer_idx = chr(65 + label)
            examples.append({
                "id": str(row.get("id", i)),
                "question": question,
                "options": option_dict,
                "answer": answer_idx,
                "label": label,
            })
        return examples

    data = {
        "train": _process(ds["train"]),
        "test": _process(ds["test"]),
    }
    _log_stats("medqa", data)
    _save_cache("medqa", data)
    return data


def load_medmcqa() -> Dict[str, List[dict]]:
    cached = _load_cache("medmcqa")
    if cached:
        return cached

    ds = load_dataset("openlifescienceai/medmcqa", trust_remote_code=True, cache_dir=CACHE_DIR)

    def _process(split, max_n=None):
        examples = []
        for i, row in enumerate(split):
            if max_n and i >= max_n:
                break
            question = row.get("question", "")
            option_dict = {
                "A": row.get("opa", ""),
                "B": row.get("opb", ""),
                "C": row.get("opc", ""),
                "D": row.get("opd", ""),
            }
            cop = row.get("cop", None)
            if cop is None or cop not in [0, 1, 2, 3]:
                continue
            answer_idx = chr(65 + cop)

            exp = row.get("exp", "")

            examples.append({
                "id": str(row.get("id", i)),
                "question": question,
                "options": option_dict,
                "explanation": exp if exp else "",
                "subject": row.get("subject_name", ""),
                "answer": answer_idx,
                "label": cop,
            })
        return examples

    train_examples = _process(ds["train"], max_n=10000)
    if "validation" in ds:
        test_examples = _process(ds["validation"], max_n=2000)
    else:
        all_ex = _process(ds["train"], max_n=12000)
        split = _stratified_split(all_ex)
        train_examples, test_examples = split["train"], split["test"]

    data = {"train": train_examples, "test": test_examples}
    _log_stats("medmcqa", data)
    _save_cache("medmcqa", data)
    return data


def load_sst2() -> Dict[str, List[dict]]:
    cached = _load_cache("sst2")
    if cached:
        return cached

    ds = load_dataset("stanfordnlp/sst2", cache_dir=CACHE_DIR)
    label_names = ["negative", "positive"]

    def _process(split, max_n=None):
        examples = []
        for i, row in enumerate(split):
            if max_n and i >= max_n:
                break
            examples.append({
                "id": str(i),
                "sentence": row["sentence"],
                "answer": label_names[row["label"]],
                "label": row["label"],
            })
        return examples

    data = {
        "train": _process(ds["train"], max_n=10000),
        "test": _process(ds["validation"]),
    }
    _log_stats("sst2", data)
    _save_cache("sst2", data)
    return data


def load_agnews() -> Dict[str, List[dict]]:
    cached = _load_cache("agnews")
    if cached:
        return cached

    ds = load_dataset("fancyzhx/ag_news", cache_dir=CACHE_DIR)
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    def _process(split, max_n=None):
        examples = []
        for i, row in enumerate(split):
            if max_n and i >= max_n:
                break
            examples.append({
                "id": str(i),
                "text": row["text"],
                "answer": label_names[row["label"]],
                "label": row["label"],
            })
        return examples

    data = {
        "train": _process(ds["train"], max_n=10000),
        "test": _process(ds["test"], max_n=2000),
    }
    _log_stats("agnews", data)
    _save_cache("agnews", data)
    return data


def load_trec() -> Dict[str, List[dict]]:
    cached = _load_cache("trec")
    if cached:
        return cached

    ds = load_dataset("trec", cache_dir=CACHE_DIR, revision="refs/convert/parquet")
    label_names = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]

    def _process(split):
        examples = []
        for i, row in enumerate(split):
            lbl = row.get("coarse_label", row.get("label-coarse", row.get("label_coarse", None)))
            if lbl is None:
                continue
            examples.append({
                "id": str(i),
                "question": row["text"],
                "answer": label_names[lbl],
                "label": lbl,
            })
        return examples

    data = {
        "train": _process(ds["train"]),
        "test": _process(ds["test"]),
    }
    _log_stats("trec", data)
    _save_cache("trec", data)
    return data

    
LOADERS = {
    "pubmedqa": load_pubmedqa,
    "medqa": load_medqa,
    "medmcqa": load_medmcqa,
    "sst2": load_sst2,
    "agnews": load_agnews,
    "trec": load_trec,
}


def load_dataset_by_name(name: str) -> Dict[str, List[dict]]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADERS.keys())}")
    return LOADERS[name]()


def format_for_embedding(example: dict, dataset_name: str) -> str:
    if dataset_name == "pubmedqa":
        return f"Question: {example['question']}\nContext: {example['context'][:500]}"
    elif dataset_name in ["medqa", "medmcqa"]:
        opts = example.get("options", {})
        opts_str = " ".join(f"({k}) {v}" for k, v in opts.items())
        return f"Question: {example['question']}\nOptions: {opts_str}"
    elif dataset_name == "sst2":
        return example["sentence"]
    elif dataset_name == "agnews":
        return example["text"][:500]
    elif dataset_name == "trec":
        return example["question"]
    else:
        return example.get("question", example.get("text", example.get("sentence", "")))


def format_example_for_embedding(example: dict) -> str:
    return format_for_embedding(example, "pubmedqa")


if __name__ == "__main__":
    for name in LOADERS:
        print(f"\n{'='*60}")
        print(f"Loading {name}...")
        data = load_dataset_by_name(name)
        print(f"  Train: {len(data['train'])}, Test: {len(data['test'])}")
        print(f"  Sample: {list(data['train'][0].keys())}")