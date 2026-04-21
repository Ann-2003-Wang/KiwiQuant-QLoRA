# 把原始数据清洗成统一的标准格式。

# 它负责：

# 读取 FinGPT-fineval、zh_fin 或你自己的 json/csv
# 清洗脏字段、空字段、重复样本
# 统一字段名
# 统一成一种标准结构
# 划分 train / val / test
# 保存到 data/processed/
import json
import random
from typing import List, Dict

from datasets import load_dataset

def load_raw_samples_from_hf(dataset_name: str, split_name: str):
    ds = load_dataset(dataset_name, split=split_name)
    return list(ds)

def load_raw_samples(file_path: str) -> List[Dict]:
    """
    支持三种格式：
    1. .jsonl: 每行一个 JSON 对象
    2. .json: 标准 JSON list / dict
    3. 特殊纯文本格式:
       0:"问题"
       1:"答案"
       0:"问题"
       1:"答案"
    """
    # ---------- 1) jsonl ----------
    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    # ---------- 2) 先尝试当标准 json 解析 ----------
    if file_path.endswith(".json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, list):
                return obj

            if isinstance(obj, dict):
                for key in ["data", "samples", "records", "items"]:
                    if key in obj and isinstance(obj[key], list):
                        return obj[key]

                # 如果本身是单个 dict，不像数据集
                raise ValueError(f"Unsupported JSON dict structure in {file_path}")

        except Exception:
            # 如果 json.load 失败，继续按特殊 0/1 文本格式解析
            pass

    # ---------- 3) 按特殊 0/1 配对文本格式解析 ----------
    samples = []
    current_question = None
    sample_id = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 只处理形如 0:"..." 或 1:"..."
            if line.startswith('0:'):
                value = line[2:].strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                current_question = value

            elif line.startswith('1:'):
                value = line[2:].strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

                if current_question is not None:
                    samples.append({
                        "id": str(sample_id),
                        "question": current_question,
                        "answer": value,
                        "source": "zh_fin"
                    })
                    sample_id += 1
                    current_question = None

    return samples

def normalize_sample(sample: dict, idx: int = 0) -> dict:
    if isinstance(sample, dict):
        question = (
            sample.get("question")
            or sample.get("input")
            or sample.get("instruction")
            or ""
        )
        answer = (
            sample.get("answer")
            or sample.get("output")
            or sample.get("response")
            or ""
        )
        return {
            "id": str(sample.get("id", idx)),
            "question": str(question).strip(),
            "answer": str(answer).strip(),
            "source": str(sample.get("source", "fingpt-fineval")),
            "task_type": "qa"
        }

    elif isinstance(sample, (list, tuple)):
        if len(sample) >= 2:
            return {
                "id": str(idx),
                "question": str(sample[0]).strip(),
                "answer": str(sample[1]).strip(),
                "source": "zh_fin",
                "task_type": "qa"
            }

    return {
        "id": str(idx),
        "question": "",
        "answer": "",
        "source": "unknown",
        "task_type": "qa"
    }

import argparse
import random

from src.config import load_yaml_config, resolve_project_paths
from src.utils import ensure_dir, save_jsonl

def filter_invalid_samples(samples):
    cleaned = []
    seen = set()
    for s in samples:
        q = s["question"].strip()
        a = s["answer"].strip()
        key = (q, a)
        if not q or not a:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    return cleaned

def split_train_val(samples, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(samples)
    n = len(samples)
    n_val = max(1, int(n * val_ratio))
    val = samples[:n_val]
    train = samples[n_val:]
    return train, val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))
    processed_dir = cfg["paths"]["processed_data"]
    ensure_dir(processed_dir)

    source_type = cfg["data"].get("source_type", "local")

    if source_type == "hf":
        dataset_name = cfg["data"]["hf_dataset_name"]
        hf_train_split = cfg["data"].get("hf_split_train", "train")
        hf_test_split = cfg["data"].get("hf_split_test", "test")

        raw_train = load_raw_samples_from_hf(dataset_name, hf_train_split)
        raw_test = load_raw_samples_from_hf(dataset_name, hf_test_split)

        train_norm = [normalize_sample(x, i) for i, x in enumerate(raw_train)]
        test_norm = [normalize_sample(x, i) for i, x in enumerate(raw_test)]

        train_clean = filter_invalid_samples(train_norm)
        test_clean = filter_invalid_samples(test_norm)

        train, val = split_train_val(
            train_clean,
            val_ratio=cfg["data"].get("val_ratio", 0.1),
            seed=cfg.get("seed", 42),
        )
        test = test_clean

        raw_count = len(raw_train) + len(raw_test)
        cleaned_count = len(train_clean) + len(test_clean)

    else:
        input_path = args.input or cfg["data"]["raw_file"]
        raw_samples = load_raw_samples(input_path)
        normalized = [normalize_sample(x, i) for i, x in enumerate(raw_samples)]
        cleaned = filter_invalid_samples(normalized)

        # 本地数据时才自己切 train/val/test
        n = len(cleaned)
        train_ratio = cfg["data"].get("train_ratio", 0.8)
        val_ratio = cfg["data"].get("val_ratio", 0.1)

        random.seed(cfg.get("seed", 42))
        random.shuffle(cleaned)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = cleaned[:n_train]
        val = cleaned[n_train:n_train+n_val]
        test = cleaned[n_train+n_val:]

        raw_count = len(raw_samples)
        cleaned_count = len(cleaned)

    save_jsonl(train, str(processed_dir / "train.jsonl"))
    save_jsonl(val, str(processed_dir / "val.jsonl"))
    save_jsonl(test, str(processed_dir / "test.jsonl"))

    print(f"raw={raw_count} cleaned={cleaned_count}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

if __name__ == "__main__":
    main()