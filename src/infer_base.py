#生成 baseline 结果
# infer_base.py

# 作用：用原始未微调模型做推理。

# 它负责：

# 加载 base model
# 读取 test 数据
# 逐条生成回答
# 保存生成结果
import torch

def generate_one(model, tokenizer, prompt: str, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def run_base_inference(model, tokenizer, infer_records):
    results = []
    for item in infer_records:
        pred = generate_one(model, tokenizer, item["prompt"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold": item["gold"],
            "prediction": pred,
            "model_name": "base_model"
        })
    return results

import argparse
from src.config import load_yaml_config, resolve_project_paths
from src.model_loader import load_tokenizer, load_base_model
from src.infer_base import run_base_inference
from src.utils import load_jsonl, save_jsonl, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))
    infer_file = cfg["paths"]["processed_data"] / "formatted" / "test_infer.jsonl"
    out_dir = cfg["paths"]["predictions"]
    ensure_dir(out_dir)

    infer_records = load_jsonl(str(infer_file))
    tokenizer = load_tokenizer(cfg["model_name"])
    model = load_base_model(cfg["model_name"], use_qlora=False)

    results = run_base_inference(model, tokenizer, infer_records)
    save_jsonl(results, str(out_dir / "base_predictions.jsonl"))

    print("base inference done:", len(results))

if __name__ == "__main__":
    main()