# 基于 QLoRA 的 Qwen2.5-7B-Instruct 金融问答微调实验

本项目围绕中文金融问答任务，使用 `FinGPT/fingpt-fineval` 数据集，对 `Qwen/Qwen2.5-7B-Instruct` 进行 QLoRA 监督微调（SFT），并完成了数据准备、训练、基线推理、微调推理、Judge 评估与结果可视化的完整实验流程。

---

## 1. 项目目标

本项目希望回答一个核心问题：

> 在金融问答任务中，针对已经较强的开源 instruct 模型，使用有限规模的领域数据进行 QLoRA 微调，是否能够显著提升模型最终回答质量？

---

## 2. 数据集

使用数据集：`FinGPT/fingpt-fineval`

数据清洗与划分结果如下：

- 原始样本数：1321
- 清洗后样本数：1307
- 训练集：939
- 验证集：104
- 测试集：264

---

## 3. 模型与方法

### 3.1 基础模型
- Backbone：`Qwen/Qwen2.5-7B-Instruct`

### 3.2 微调方式
- 方法：QLoRA / SFT
- 量化：4-bit NF4
- LoRA 目标模块：
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`

### 3.3 训练配置（最终版本）
- `max_seq_length = 1024`
- `per_device_train_batch_size = 2`
- `per_device_eval_batch_size = 2`
- `gradient_accumulation_steps = 8`
- `learning_rate = 0.0001`
- `num_train_epochs = 1`
- `logging_steps = 5`
- `eval_steps = 10`
- `save_steps = 50`
- `lora_r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`

---

## 4. 项目结构

```text
finance_lora_sft/
├── configs/
│   ├── train_default.yaml
│   └── eval_default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── adapters/
│   ├── predictions/
│   ├── eval/
│   └── figures/
├── src/
│   ├── config.py
│   ├── data_prepare.py
│   ├── dataset_builder.py
│   ├── model_loader.py
│   ├── train_sft.py
│   ├── infer_base.py
│   ├── infer_sft.py
│   ├── judge_eval.py
│   ├── metrics.py
│   └── plot_results.py
└── README.md
```

---

## 5. 实验流程

### 5.1 数据准备
```bash
python -m src.data_prepare --config configs/train_default.yaml
```

### 5.2 构造训练/推理文本
```bash
python -m src.dataset_builder --config configs/train_default.yaml --mode sft --split train
python -m src.dataset_builder --config configs/train_default.yaml --mode sft --split val
python -m src.dataset_builder --config configs/train_default.yaml --mode infer --split test
```

### 5.3 训练
```bash
python -m src.train_sft --config configs/train_default.yaml
```

### 5.4 基线模型推理
```bash
python -m src.infer_base --config configs/eval_default.yaml
```

### 5.5 微调模型推理
```bash
python -m src.infer_sft --config configs/eval_default.yaml
```

### 5.6 Judge 评估
```bash
export DEEPSEEK_API_KEY="YOUR_KEY"
export JUDGE_BASE_URL="https://api.deepseek.com"

python -m src.judge_eval --config configs/eval_default.yaml
```

### 5.7 汇总指标
```bash
python -m src.metrics --config configs/eval_default.yaml
```

### 5.8 结果可视化
```bash
python -m src.plot_results --config configs/eval_default.yaml --trainer_log outputs/adapters/qwen25_7b_finance_lora/trainer_state.json
```

---

## 6. 训练结果

本轮训练总步数约为 59，训练过程中的 loss 变化如下：

- train loss 从约 `2.48` 逐步下降到末期约 `0.92`
- eval loss 从约 `1.89` 逐步下降到约 `0.93`
- 曲线整体表现为：**前半段下降较快，后半段趋于平稳**

这说明模型已经学到金融问答数据分布，并且训练过程是稳定收敛的。

---

## 7. 评估结果

在测试集 264 条样本上，使用 DeepSeek 作为 judge 进行 Base vs SFT pairwise 评估，得到结果：

- Base 平均分：`8.2348`
- SFT 平均分：`7.4848`
- Base 胜率：`0.5492`
- SFT 胜率：`0.1705`
- Tie 比例：`0.2803`
- 样本数：`264`

### 结果解读
当前实验表明：

1. 微调后的模型确实学到了训练集分布；
2. 但在整体回答质量上，当前 SFT 版本尚未超过原始 baseline；
3. 说明对于能力较强的 instruct 模型，有限规模领域 SFT 并不保证带来整体性能提升。

---

## 8. 可视化结果

项目已生成如下图表：

- `outputs/figures/loss_curve.png`
- `outputs/figures/score_bar.png`

### 图像解读
- `loss_curve.png`：训练损失与验证损失均下降，前期下降明显，后期逐步趋于平稳；
- `score_bar.png`：Base 的平均得分高于 SFT，反映出当前微调版本整体仍弱于 baseline。

---

## 9. 案例分析（Case Study）

### 9.1 共同失败型
在部分强事实约束的金融客观题中，Base 与 SFT 会同时给出错误答案。例如某些“概念-指标对应”题，两个模型都选错了同一个选项，说明当前微调尚未显著提升这类知识性判断题的稳定性。

### 9.2 Base 因解释完整而占优
在不少样本中，Base 与 SFT 都能给出正确选项，但 Base 会进一步解释其他选项为什么错误，因此在 judge 看来更完整、更专业。例如无形资产摊销、政府会计、审计程序不可预见性等题目都体现了这一点。

### 9.3 Base 在计算题上更稳定
在涉及会计或财务数值推导的题目中，Base 往往能给出完整计算过程并得出正确答案，而 SFT 有时会直接给出错误选项，说明当前微调未能提升此类结构化推导能力。

### 9.4 SFT 在部分客观题上能纠正 Base 错误
尽管整体结果不如 Base，但 SFT 仍在部分题目上胜出。例如在期货交易真正目的、可转换债券差额计入所有者权益等题目中，SFT 直接给出正确答案，而 Base 出现了结论错误或分析自相矛盾的情况。

### 9.5 Judge 评估存在一定噪声
个别样本中，judge 的偏好与标准答案并不完全一致，说明基于大模型的自动评估虽然高效，但仍可能存在噪声，因此结果分析应结合人工案例审阅。

---

## 10. 目前结论

本项目已经完成了一个完整的金融问答 QLoRA 实验闭环。当前结果说明：

- 模型训练稳定；
- 验证损失下降明显；
- 但整体评估中 SFT 尚未超过原始 baseline。

因此，本实验的关键发现是：

> 在强 baseline 的基础上，有限规模金融领域数据上的 QLoRA 微调并不必然提升最终回答质量。数据风格、提示模板、解释完整性与训练超参数都会显著影响最终表现。

---

## 11. 后续优化方向

后续可继续从以下方向优化：

1. 调整学习率和训练轮数；
2. 进一步优化 instruction + question 的拼接方式；
3. 尝试更小或更新版本的 backbone；
4. 对 Base / SFT / Tie 三类样本分别做更细致的 case study；
5. 将 LLM judge 与规则评估（如选项匹配）结合，减轻评估噪声。
