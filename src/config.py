# 作用：统一读取配置。

# 它负责：
# 读取 yaml 配置文件
# 组织路径
# 读取环境变量
# 合并默认参数和命令行参数

# 为什么要有它：
# 因为以后你可能会换模型、换 batch size、换输出目录。没有配置层的话，你就会像现在 notebook 一样到处改字符串。

# 大概负责这些配置：

# 模型名
# 数据路径
# 最大长度
# LoRA 参数
# 训练参数
# 输出路径
# judge API 的模型名
#这个文件一般不需要单独跑，但为了调试方便，可以给一个 main()：

from pathlib import Path
import os
import yaml

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_project_paths(cfg: dict) -> dict:
    root = Path(cfg.get("project_root", ".")).resolve()
    cfg["paths"] = {
        "root": root,
        "raw_data": root / "data" / "raw",
        "processed_data": root / "data" / "processed",
        "outputs": root / "outputs",
        "predictions": root / "outputs" / "predictions",
        "eval": root / "outputs" / "eval",
        "figures": root / "outputs" / "figures",
        "checkpoints": root / "outputs" / "checkpoints",
        "adapters": root / "outputs" / "adapters",
    }
    return cfg

def load_env_secrets() -> dict:
    return {
        "judge_api_key": os.getenv("JUDGE_API_KEY"),
        "judge_base_url": os.getenv("JUDGE_BASE_URL"),
    }

import argparse
from src.config import load_yaml_config, resolve_project_paths, load_env_secrets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cfg = resolve_project_paths(cfg)
    secrets = load_env_secrets()

    print("model_name:", cfg.get("model_name"))
    print("paths:", cfg["paths"])
    print("judge_api_key_exists:", bool(secrets.get("judge_api_key")))

if __name__ == "__main__":
    main()