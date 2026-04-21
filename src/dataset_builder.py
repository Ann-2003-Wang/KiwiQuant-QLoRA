from typing import Dict, List
import argparse

from src.config import load_yaml_config, resolve_project_paths
from src.model_loader import load_tokenizer
from src.utils import load_jsonl, save_jsonl, ensure_dir

SYSTEM_PROMPT = "你是一名专业的金融知识助手，请简洁、准确地回答用户问题。"


def build_user_content(sample: Dict) -> str:
    question = sample.get("question", "").strip()
    instruction = sample.get("instruction", "").strip()
    if instruction:
        return f"{instruction}\n\n{question}"
    return question


def build_messages(sample: Dict) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(sample)},
        {"role": "assistant", "content": sample["answer"]},
    ]


def format_for_sft(sample: Dict, tokenizer) -> Dict:
    messages = build_messages(sample)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {
        "id": sample["id"],
        "text": text,
        "question": sample["question"],
        "answer": sample["answer"],
        "instruction": sample.get("instruction", ""),
    }


def format_for_inference(sample: Dict, tokenizer) -> Dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(sample)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {
        "id": sample["id"],
        "prompt": prompt,
        "question": sample["question"],
        "gold": sample["answer"],
        "instruction": sample.get("instruction", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["sft", "infer"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))
    model_cfg = cfg.get("model", {})

    tokenizer = load_tokenizer(
        cfg["model_name"],
        cache_dir=model_cfg.get("cache_dir"),
        local_files_only=model_cfg.get("local_files_only", False),
    )

    processed_dir = cfg["paths"]["processed_data"]
    out_dir = cfg["paths"]["processed_data"] / "formatted"
    ensure_dir(out_dir)

    records = load_jsonl(str(processed_dir / f"{args.split}.jsonl"))

    if args.mode == "sft":
        formatted = [format_for_sft(x, tokenizer) for x in records]
        save_jsonl(formatted, str(out_dir / f"{args.split}_sft.jsonl"))
    else:
        formatted = [format_for_inference(x, tokenizer) for x in records]
        save_jsonl(formatted, str(out_dir / f"{args.split}_infer.jsonl"))

    print(f"saved {len(formatted)} records to {out_dir}")


if __name__ == "__main__":
    main()