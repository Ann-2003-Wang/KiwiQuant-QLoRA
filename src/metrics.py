# 作用：做一些不依赖 judge 的基础指标统计。

# 它负责：

# 统计平均分
# 统计胜率
# 统计回答长度
# 统计格式正确率
# 可选地做简单文本相似度

# 为什么需要它：
# 因为不是每次都要重新调用 API。很多汇总统计完全可以本地做。
def compute_average_scores(judge_records):
    score_a = [x["score_a"] for x in judge_records if x["score_a"] is not None]
    score_b = [x["score_b"] for x in judge_records if x["score_b"] is not None]
    return {
        "avg_base": sum(score_a) / len(score_a) if score_a else None,
        "avg_sft": sum(score_b) / len(score_b) if score_b else None,
    }

def compute_win_rate(judge_records):
    total = len(judge_records)
    base_win = sum(1 for x in judge_records if x["winner"] == "base")
    sft_win = sum(1 for x in judge_records if x["winner"] == "sft")
    tie = sum(1 for x in judge_records if x["winner"] == "tie")
    return {
        "base_win_rate": base_win / total if total else 0,
        "sft_win_rate": sft_win / total if total else 0,
        "tie_rate": tie / total if total else 0,
    }

def summarize_eval(judge_records):
    summary = {}
    summary.update(compute_average_scores(judge_records))
    summary.update(compute_win_rate(judge_records))
    summary["n_samples"] = len(judge_records)
    return summary

import argparse
import json
from src.config import load_yaml_config, resolve_project_paths
from src.metrics import summarize_eval
from src.utils import load_jsonl, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))
    eval_dir = cfg["paths"]["eval"]
    ensure_dir(eval_dir)

    judge_records = load_jsonl(str(eval_dir / "judge_records.jsonl"))
    summary = summarize_eval(judge_records)

    with open(eval_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(summary)

if __name__ == "__main__":
    main()
    #统计平均分、胜率、样本数。