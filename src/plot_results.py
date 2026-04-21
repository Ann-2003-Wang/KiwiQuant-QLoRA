import argparse
import json
import matplotlib.pyplot as plt

from src.config import load_yaml_config, resolve_project_paths
from src.utils import ensure_dir


def plot_loss_curve(log_history, save_path: str):
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    current_step = 0

    for x in log_history:
        if "step" in x:
            current_step = x["step"]

        if "loss" in x and "eval_loss" not in x:
            train_steps.append(current_step)
            train_loss.append(float(x["loss"]))

        if "eval_loss" in x:
            eval_steps.append(current_step)
            eval_loss.append(float(x["eval_loss"]))

    plt.figure(figsize=(8, 5))

    if train_steps and train_loss:
        plt.plot(train_steps, train_loss, label="train_loss")

    if eval_steps and eval_loss:
        plt.plot(eval_steps, eval_loss, label="eval_loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_score_bar(summary: dict, save_path: str):
    labels = ["Base", "SFT"]
    values = [summary["avg_base"], summary["avg_sft"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trainer_log", type=str, default=None)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))
    fig_dir = cfg["paths"]["figures"]
    eval_dir = cfg["paths"]["eval"]
    ensure_dir(fig_dir)

    trainer_log_path = args.trainer_log or str(
        cfg["paths"]["adapters"] / cfg["experiment_name"] / "trainer_state.json"
    )

    with open(trainer_log_path, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    log_history = trainer_state["log_history"]

    with open(eval_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    plot_loss_curve(log_history, str(fig_dir / "loss_curve.png"))
    plot_score_bar(summary, str(fig_dir / "score_bar.png"))

    print("figures saved to:", fig_dir)


if __name__ == "__main__":
    main()