import argparse
import json
import os
import re
import random
from openai import OpenAI

from src.config import load_yaml_config, resolve_project_paths
from src.utils import load_jsonl, save_jsonl, ensure_dir


def build_judge_prompt(question: str, gold: str, answer_a: str, answer_b: str) -> str:
    return f"""
你是一名严格的金融问答评审员。请比较两个回答的准确性、专业性、完整性，并参考标准答案进行判断。

评分标准：
1. 准确性：是否正确回答问题，是否有事实错误
2. 专业性：是否使用了恰当的金融概念和表达
3. 完整性：是否覆盖了关键点，是否过于简略或遗漏核心信息

问题：{question}
参考答案：{gold}

回答A：{answer_a}

回答B：{answer_b}

请只输出一个 JSON 对象，不要输出其他文字，不要使用 Markdown 代码块。
格式如下：
{{
  "winner": "A",
  "score_a": 8,
  "score_b": 6,
  "reason": "A 更准确且覆盖了关键点。"
}}

其中：
- "winner" 只能是 "A"、"B" 或 "Tie"
- "score_a" 和 "score_b" 是 0 到 10 的整数
- "reason" 是一句简短中文说明
""".strip()


def safe_extract_json(text: str) -> dict:
    """
    尽量从模型输出中提取 JSON。
    兼容：
    - 直接输出 JSON
    - ```json ... ```
    - 前后夹杂说明文字
    """
    text = text.strip()

    # 1) 直接尝试整体解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 去掉 markdown 代码块
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # 3) 提取第一个大括号对象
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {
        "winner": "Tie",
        "score_a": None,
        "score_b": None,
        "reason": "JSON parse failed"
    }


def normalize_judge_result(parsed: dict) -> dict:
    winner = str(parsed.get("winner", "Tie")).strip()
    if winner not in {"A", "B", "Tie"}:
        winner = "Tie"

    def to_int_or_none(x):
        try:
            v = int(x)
            if 0 <= v <= 10:
                return v
            return None
        except Exception:
            return None

    score_a = to_int_or_none(parsed.get("score_a"))
    score_b = to_int_or_none(parsed.get("score_b"))
    reason = str(parsed.get("reason", "")).strip() or "No reason provided"

    return {
        "winner": winner,
        "score_a": score_a,
        "score_b": score_b,
        "reason": reason,
    }


def parse_judge_output(text: str) -> dict:
    parsed = safe_extract_json(text)
    return normalize_judge_result(parsed)


def build_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def deepseek_client_fn(client, prompt: str, model_name: str = "deepseek-chat") -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def run_pairwise_judge(client, base_results, sft_results, judge_model: str = "deepseek-chat"):
    merged = []

    # 用 id 对齐，避免 zip 顺序错位
    sft_map = {x["id"]: x for x in sft_results}

    for base_item in base_results:
        sample_id = base_item["id"]
        if sample_id not in sft_map:
            continue

        sft_item = sft_map[sample_id]

        question = base_item["question"]
        gold = base_item["gold"]
        base_answer = base_item["prediction"]
        sft_answer = sft_item["prediction"]

        # 随机交换 A/B，减少位置偏差
        swap = random.choice([True, False])
        if swap:
            answer_a, answer_b = sft_answer, base_answer
            mapping = {"A": "sft", "B": "base", "Tie": "tie"}
        else:
            answer_a, answer_b = base_answer, sft_answer
            mapping = {"A": "base", "B": "sft", "Tie": "tie"}

        prompt = build_judge_prompt(question, gold, answer_a, answer_b)
        raw = deepseek_client_fn(client, prompt, model_name=judge_model)
        parsed = parse_judge_output(raw)

        winner_raw = parsed["winner"]
        winner = mapping.get(winner_raw, "tie")

        merged.append({
            "id": sample_id,
            "question": question,
            "gold": gold,
            "base_answer": base_answer,
            "sft_answer": sft_answer,
            "winner": winner,
            "winner_raw": winner_raw,
            "score_a": parsed["score_a"],
            "score_b": parsed["score_b"],
            "reason": parsed["reason"],
            "swap_ab": swap,
            "raw_judge_output": raw,
        })

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))

    pred_dir = cfg["paths"]["predictions"]
    eval_dir = cfg["paths"]["eval"]
    ensure_dir(eval_dir)

    base_file = pred_dir / "base_predictions.jsonl"
    sft_file = pred_dir / "sft_predictions.jsonl"

    base_results = load_jsonl(str(base_file))
    sft_results = load_jsonl(str(sft_file))

    judge_model = cfg.get("eval", {}).get("judge_model", "deepseek-chat")

    client = build_deepseek_client()
    judge_records = run_pairwise_judge(
        client=client,
        base_results=base_results,
        sft_results=sft_results,
        judge_model=judge_model,
    )

    save_jsonl(judge_records, str(eval_dir / "judge_records.jsonl"))

    print("judge evaluation done:", len(judge_records))
    print("saved to:", eval_dir / "judge_records.jsonl")


if __name__ == "__main__":
    main()