import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

TASK_CONFIGS = {
    "pubmedqa": {
        "system": (
            "You are a biomedical expert. Given a medical research question and context "
            "from a PubMed abstract, answer with exactly one word: yes, no, or maybe."
        ),
        "demo_template": "Question: {question}\nContext: {context}\nAnswer: {answer}",
        "query_template": "Question: {question}\nContext: {context}\nAnswer:",
        "labels": ["yes", "no", "maybe"],
        "field_map": {"context_field": "context", "max_context": 300},
    },
    "medqa": {
        "system": (
            "You are a medical expert. Given a clinical question and four options, "
            "respond with exactly one letter: A, B, C, or D."
        ),
        "demo_template": "Question: {question}\n(A) {opt_a}\n(B) {opt_b}\n(C) {opt_c}\n(D) {opt_d}\nAnswer: {answer}",
        "query_template": "Question: {question}\n(A) {opt_a}\n(B) {opt_b}\n(C) {opt_c}\n(D) {opt_d}\nAnswer:",
        "labels": ["A", "B", "C", "D"],
        "field_map": {},
    },
    "medmcqa": {
        "system": (
            "You are a medical expert. Given a medical question and four options, "
            "respond with exactly one letter: A, B, C, or D."
        ),
        "demo_template": "Question: {question}\n(A) {opt_a}\n(B) {opt_b}\n(C) {opt_c}\n(D) {opt_d}\nAnswer: {answer}",
        "query_template": "Question: {question}\n(A) {opt_a}\n(B) {opt_b}\n(C) {opt_c}\n(D) {opt_d}\nAnswer:",
        "labels": ["A", "B", "C", "D"],
        "field_map": {},
    },
    "sst2": {
        "system": (
            "Classify the sentiment of the following sentence as exactly one word: "
            "positive or negative."
        ),
        "demo_template": "Sentence: {sentence}\nSentiment: {answer}",
        "query_template": "Sentence: {sentence}\nSentiment:",
        "labels": ["negative", "positive"],
        "field_map": {},
    },
    "agnews": {
        "system": (
            "Classify the following news article into exactly one category: "
            "World, Sports, Business, or Sci/Tech."
        ),
        "demo_template": "Article: {text}\nCategory: {answer}",
        "query_template": "Article: {text}\nCategory:",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "field_map": {"max_text": 300},
    },
    "trec": {
        "system": (
            "Classify the following question into exactly one type: "
            "ABBR (abbreviation), DESC (description), ENTY (entity), "
            "HUM (human), LOC (location), or NUM (numeric). "
            "Respond with exactly one word."
        ),
        "demo_template": "Question: {question}\nType: {answer}",
        "query_template": "Question: {question}\nType:",
        "labels": ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
        "field_map": {},
    },
}

# Separator inserted between the demonstrations and the test question
# so the model clearly knows when examples end and the real query begins.
DEMO_QUERY_SEPARATOR = (
    "\n\n---\n"
    "Now answer the following. Respond with ONLY the answer, nothing else.\n\n"
)


def _fill_template(template: str, example: dict, dataset_name: str) -> str:
    cfg = TASK_CONFIGS[dataset_name]
    fields = {}

    if dataset_name == "pubmedqa":
        max_ctx = cfg["field_map"].get("max_context", 300)
        fields = {
            "question": example["question"],
            "context": example["context"][:max_ctx],
            "answer": example.get("answer", ""),
        }
    elif dataset_name in ["medqa", "medmcqa"]:
        opts = example.get("options", {})
        fields = {
            "question": example["question"],
            "opt_a": opts.get("A", ""),
            "opt_b": opts.get("B", ""),
            "opt_c": opts.get("C", ""),
            "opt_d": opts.get("D", ""),
            "answer": example.get("answer", ""),
        }
    elif dataset_name == "sst2":
        fields = {
            "sentence": example["sentence"],
            "answer": example.get("answer", ""),
        }
    elif dataset_name == "agnews":
        max_text = cfg["field_map"].get("max_text", 300)
        fields = {
            "text": example["text"][:max_text],
            "answer": example.get("answer", ""),
        }
    elif dataset_name == "trec":
        fields = {
            "question": example["question"],
            "answer": example.get("answer", ""),
        }

    return template.format(**fields)


def build_prompt(
    demonstrations: List[dict],
    query: dict,
    dataset_name: str,
) -> List[Dict[str, str]]:
    """
    Build a chat-formatted prompt with demonstrations and a test query.

    Uses a clear separator between demonstrations and the query so the model
    knows when the examples end and the real question begins.
    """
    cfg = TASK_CONFIGS[dataset_name]
    messages = [{"role": "system", "content": cfg["system"]}]

    demo_blocks = [_fill_template(cfg["demo_template"], d, dataset_name) for d in demonstrations]
    query_block = _fill_template(cfg["query_template"], query, dataset_name)

    if demo_blocks:
        # Demos joined together, then a clear separator, then the query
        demos_text = "\n\n".join(demo_blocks)
        full_content = demos_text + DEMO_QUERY_SEPARATOR + query_block
    else:
        # Zero-shot: just the query
        full_content = query_block

    messages.append({"role": "user", "content": full_content})
    return messages


def build_prompt_with_budget(
    demonstrations: List[dict],
    query: dict,
    dataset_name: str,
    tokenizer,
    max_input_tokens: int,
) -> List[Dict[str, str]]:
    """
    Build a prompt that is guaranteed to fit within max_input_tokens.

    Reserves space for the system message, separator, and query first,
    then greedily adds as many demonstrations as will fit.

    Args:
        demonstrations: Candidate demo examples (in priority order).
        query: The test example to classify.
        dataset_name: Which dataset/task config to use.
        tokenizer: The model's tokenizer (for counting tokens).
        max_input_tokens: Hard limit on total input tokens.

    Returns:
        Chat messages list that fits within the budget.
    """
    cfg = TASK_CONFIGS[dataset_name]
    query_block = _fill_template(cfg["query_template"], query, dataset_name)

    zero_shot_messages = [
        {"role": "system", "content": cfg["system"]},
        {"role": "user", "content": DEMO_QUERY_SEPARATOR + query_block},
    ]
    try:
        zero_shot_prompt = tokenizer.apply_chat_template(
            zero_shot_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        zero_shot_prompt = "\n".join(m["content"] for m in zero_shot_messages)

    fixed_tokens = len(tokenizer.encode(zero_shot_prompt))
    remaining_budget = max_input_tokens - fixed_tokens

    if remaining_budget <= 0:
        logger.warning(
            f"Query alone uses {fixed_tokens} tokens (budget={max_input_tokens}). "
            f"Zero-shot fallback."
        )
        return build_prompt([], query, dataset_name)

    selected_demos = []
    tokens_used = 0
    for demo in demonstrations:
        demo_text = _fill_template(cfg["demo_template"], demo, dataset_name) + "\n\n"
        demo_tokens = len(tokenizer.encode(demo_text))
        if tokens_used + demo_tokens > remaining_budget:
            break
        selected_demos.append(demo)
        tokens_used += demo_tokens

    if len(selected_demos) < len(demonstrations):
        logger.info(
            f"Demo budget: kept {len(selected_demos)}/{len(demonstrations)} demos "
            f"({tokens_used}/{remaining_budget} tokens used)"
        )

    return build_prompt(selected_demos, query, dataset_name)


def build_pubmedqa_prompt(
    demonstrations: List[dict],
    query: dict,
    max_context_len: int = 300,
) -> List[Dict[str, str]]:
    return build_prompt(demonstrations, query, "pubmedqa")


def parse_response(response: str, dataset_name: str) -> str:
    labels = TASK_CONFIGS[dataset_name]["labels"]
    response = response.strip()
    response_lower = response.lower()

    labels_lower = [l.lower() for l in labels]

    for label, label_l in zip(labels, labels_lower):
        if response_lower == label_l or response_lower == f"{label_l}.":
            return label

    first_word = response.split()[0].strip(".,!?:()") if response else ""
    first_word_lower = first_word.lower()
    for label, label_l in zip(labels, labels_lower):
        if first_word_lower == label_l:
            return label

    if dataset_name in ["medqa", "medmcqa"]:
        for char in response.upper():
            if char in ["A", "B", "C", "D"]:
                return char

    for label, label_l in zip(labels, labels_lower):
        if label_l in response_lower:
            return label

    return "unknown"


def parse_pubmedqa_response(response: str) -> str:
    return parse_response(response, "pubmedqa")