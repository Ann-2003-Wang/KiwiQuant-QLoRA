 #环境
export DEEPSEEK_API_KEY="sk-ed23ef1a37134535a8d93be537a575d0"
export JUDGE_BASE_URL="https://api.deepseek.com"

#1. 进入环境和项目目录
conda activate qwen-lora
cd /225040103/finance_lora_sft
#2. 数据准备
python -m src.data_prepare --config configs/train_default.yaml
#3. 构造训练/验证/测试格式化数据
python -m src.dataset_builder --config configs/train_default.yaml --mode sft --split train
python -m src.dataset_builder --config configs/train_default.yaml --mode sft --split val
python -m src.dataset_builder --config configs/train_default.yaml --mode infer --split test
#4. 第一轮训练
python -m src.train_sft --config configs/train_default.yaml
#5. baseline 推理
python -m src.infer_base --config configs/eval_default.yaml
#6. sft 推理
python -m src.infer_sft --config configs/eval_default.yaml
#7. 设置 DeepSeek 环境变量
export DEEPSEEK_API_KEY="你的key"
export JUDGE_BASE_URL="https://api.deepseek.com"
#8. judge 评估
python -m src.judge_eval --config configs/eval_default.yaml
#9. 汇总指标
python -m src.metrics --config configs/eval_default.yaml
#10. 查找 trainer_state.json
find outputs/adapters/qwen25_7b_finance_lora -name "trainer_state.json"
#11. 用正确 checkpoint 路径画图
python -m src.plot_results --config configs/eval_default.yaml --trainer_log outputs/adapters/qwen25_7b_finance_lora/checkpoint-118/trainer_state.json
#12. 检查图是否生成
ls outputs/figures