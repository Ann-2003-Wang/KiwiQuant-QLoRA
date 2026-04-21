# 作用：用 LoRA 微调后的模型做推理。

# 它负责：

# 加载 base model + adapter
# 读取 test 数据
# 生成回答
# 保存结果

# 和 infer_base.py 很像，但加载方式不同。
# 把它单独拆出来的好处是：
# base 和 sft 两条推理线互不影响，比较清楚。
from peft import PeftModel
from src.infer_base import generate_one
def load_sft_model(base_model, adapter_path: str):
    return PeftModel.from_pretrained(base_model, adapter_path)

def run_sft_inference(model, tokenizer, infer_records):
    results = []
    for item in infer_records:
        pred = generate_one(model, tokenizer, item["prompt"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold": item["gold"],
            "prediction": pred,
            "model_name": "sft_model"
        })
    return results

import argparse
from src.config import load_yaml_config, resolve_project_paths
from src.model_loader import load_tokenizer, load_base_model
from src.infer_sft import load_sft_model, run_sft_inference
from src.utils import load_jsonl, save_jsonl, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))
    infer_file = cfg["paths"]["processed_data"] / "formatted" / "test_infer.jsonl"
    out_dir = cfg["paths"]["predictions"]
    ensure_dir(out_dir)

    adapter_path = args.adapter_path or str(cfg["paths"]["adapters"] / cfg["experiment_name"])
    infer_records = load_jsonl(str(infer_file))

    tokenizer = load_tokenizer(cfg["model_name"])
    base_model = load_base_model(cfg["model_name"], use_qlora=False)
    model = load_sft_model(base_model, adapter_path)

    results = run_sft_inference(model, tokenizer, infer_records)
    save_jsonl(results, str(out_dir / "sft_predictions.jsonl"))

    print("sft inference done:", len(results))
    print("adapter used:", adapter_path)

if __name__ == "__main__":
    main()
    
# 作用：
# 加载 adapter，生成微调后结果。