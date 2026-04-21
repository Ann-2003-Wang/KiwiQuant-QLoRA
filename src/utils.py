# 作用：放通用小工具。

# 它负责：

# 设置随机种子
# 创建目录
# 保存 json/jsonl
# 读取 json/jsonl
# 简单日志函数
# 文本清洗小函数
import os
import json
import random
import numpy as np
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_jsonl(records, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(file_path: str):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
#这个文件也不用单独跑，但可以保留一个 smoke test：
import argparse
from src.utils import ensure_dir, save_jsonl, load_jsonl, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_dir", type=str, default="outputs/tmp")
    args = parser.parse_args()

    ensure_dir(args.tmp_dir)
    set_seed(42)

    sample = [{"x": 1}, {"x": 2}]
    path = f"{args.tmp_dir}/tmp.jsonl"
    save_jsonl(sample, path)
    loaded = load_jsonl(path)

    print("loaded:", loaded)

if __name__ == "__main__":
    main()