import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from config import InferenceConfig, RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INFERENCE_CACHE_DIR = RESULTS_DIR / "inference_cache"
INFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class LocalLLM:
    def __init__(self, cfg: InferenceConfig = InferenceConfig()):
        self.cfg = cfg
        self.device = cfg.device
        logger.info(f"Loading model: {cfg.model_name} on {self.device}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            padding_side="left",
            cache_dir=cfg.cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cfg.cache_dir,
        )
        self.model.eval()
        self.model_short = cfg.model_name.split("/")[-1]

        self.max_context_length = cfg.max_context_length
        logger.info(
            f"Model loaded: {self.model_short} | "
            f"max_context={self.max_context_length} tokens"
        )

        self._truncation_count = 0
        self._generation_count = 0

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        max_input_len = self.max_context_length - max_tokens

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(
                f"{'### ' if m['role'] == 'system' else ''}{m['content']}"
                for m in messages
            ) + "\n"

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        original_len = len(input_ids)

        self._generation_count += 1

        if original_len > max_input_len:
            input_ids = input_ids[-max_input_len:]
            self._truncation_count += 1
            logger.warning(
                f"Prompt truncated: {original_len} -> {max_input_len} tokens "
                f"(LEFT-truncated, dropped {original_len - max_input_len} leading tokens) "
                f"[{self._truncation_count}/{self._generation_count} prompts truncated]"
            )

        inputs = self.tokenizer(
            self.tokenizer.decode(input_ids, skip_special_tokens=False),
            return_tensors="pt",
            truncation=False,  # already handled above
            add_special_tokens=False,  # already in the decoded string
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
            do_sample=self.cfg.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def get_truncation_stats(self) -> Dict:
        return {
            "total_generations": self._generation_count,
            "truncated": self._truncation_count,
            "truncation_rate": (
                self._truncation_count / self._generation_count
                if self._generation_count > 0 else 0.0
            ),
        }


def run_inference(
    llm: LocalLLM,
    prompts: List[List[Dict[str, str]]],
    cache_name: str = "default",
) -> List[str]:
    cache_path = INFERENCE_CACHE_DIR / f"{cache_name}_{llm.model_short}.json"

    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        logger.info(f"Loaded {len(cache)} cached responses from {cache_path}")

    responses = []
    new_count = 0

    for prompt in tqdm(prompts, desc=f"Inference ({llm.model_short})"):
        prompt_hash = hashlib.md5(json.dumps(prompt).encode()).hexdigest()

        if prompt_hash in cache:
            responses.append(cache[prompt_hash])
        else:
            response = llm.generate(prompt)
            cache[prompt_hash] = response
            responses.append(response)
            new_count += 1

            if new_count % 10 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    stats = llm.get_truncation_stats()
    logger.info(
        f"Inference complete: {new_count} new, {len(responses) - new_count} cached | "
        f"Truncation: {stats['truncated']}/{stats['total_generations']} "
        f"({stats['truncation_rate']:.1%})"
    )

    return responses