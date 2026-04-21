# 只负责加载模型和 tokenizer。

# 它负责：

# 加载 Qwen tokenizer
# 加载 base model
# 配置 4-bit quantization
# 配置 BF16
# 配置 device map
# 应用 LoRA 配置

# 为什么要单独拆：
# 以后你如果从 Qwen2.5-7B-Instruct 换成 Qwen3-8B，只需要改这里或改配置，不会影响训练、推理、评估脚本。
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def build_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(model_name: str, cache_dir: str = None, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    use_qlora: bool = True,
    cache_dir: str = None,
    local_files_only: bool = False,
):
    common_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "device_map": "auto",
    }

    if use_qlora:
        bnb_config = build_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            **common_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            **common_kwargs,
        )

    return model


def attach_lora(model, r=16, alpha=32, dropout=0.05):
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model


# 下面只是可选测试入口
if __name__ == "__main__":
    import argparse
    from src.config import load_yaml_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--with_lora", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg.get("model", {})

    tokenizer = load_tokenizer(
        cfg["model_name"],
        cache_dir=model_cfg.get("cache_dir"),
        local_files_only=model_cfg.get("local_files_only", False),
    )

    model = load_base_model(
        cfg["model_name"],
        use_qlora=cfg["train"].get("use_qlora", True),
        cache_dir=model_cfg.get("cache_dir"),
        local_files_only=model_cfg.get("local_files_only", False),
    )

    if args.with_lora:
        model = attach_lora(
            model,
            r=cfg["train"].get("lora_r", 16),
            alpha=cfg["train"].get("lora_alpha", 32),
            dropout=cfg["train"].get("lora_dropout", 0.05),
        )

    print("tokenizer loaded:", tokenizer.__class__.__name__)
    print("model loaded:", model.__class__.__name__)
    
    #这个文件通常不单独跑，但你可以加一个最小测试入口