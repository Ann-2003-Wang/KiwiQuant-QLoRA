import argparse
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from src.config import load_yaml_config, resolve_project_paths
from src.model_loader import load_tokenizer, load_base_model, attach_lora
from src.utils import load_jsonl, ensure_dir, set_seed


def build_hf_dataset(records):
    return Dataset.from_list(records)


def build_training_args(output_dir: str, cfg: dict):
    tcfg = cfg["train"]
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        num_train_epochs=tcfg["num_train_epochs"],

        logging_strategy=tcfg["logging_strategy"],
        logging_steps=tcfg["logging_steps"],

        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],

        save_strategy=tcfg["save_strategy"],
        save_steps=tcfg["save_steps"],

        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=tcfg["warmup_ratio"],
        save_total_limit=tcfg["save_total_limit"],
        report_to="none",

        dataset_text_field="text",
        max_length=tcfg.get("max_seq_length", 1024),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = resolve_project_paths(load_yaml_config(args.config))
    set_seed(cfg.get("seed", 42))

    model_cfg = cfg.get("model", {})
    train_cfg = cfg["train"]

    train_file = cfg["paths"]["processed_data"] / "formatted" / "train_sft.jsonl"
    val_file = cfg["paths"]["processed_data"] / "formatted" / "val_sft.jsonl"
    output_dir = cfg["paths"]["adapters"] / cfg["experiment_name"]
    ensure_dir(output_dir)

    train_records = load_jsonl(str(train_file))
    val_records = load_jsonl(str(val_file))

    train_ds = build_hf_dataset(train_records)
    val_ds = build_hf_dataset(val_records)

    tokenizer = load_tokenizer(
        cfg["model_name"],
        cache_dir=model_cfg.get("cache_dir"),
        local_files_only=model_cfg.get("local_files_only", False),
    )

    model = load_base_model(
        cfg["model_name"],
        use_qlora=train_cfg.get("use_qlora", True),
        cache_dir=model_cfg.get("cache_dir"),
        local_files_only=model_cfg.get("local_files_only", False),
    )

    model = attach_lora(
        model,
        r=train_cfg["lora_r"],
        alpha=train_cfg["lora_alpha"],
        dropout=train_cfg["lora_dropout"],
    )

    training_args = build_training_args(str(output_dir), cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    trainer.save_state()

    print("training finished")
    print("adapter saved to:", output_dir)


if __name__ == "__main__":
    main()