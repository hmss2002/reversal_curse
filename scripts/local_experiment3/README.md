# Local Experiment 3: Reversing Instructions

本地开源模型复现 Experiment 3（反转指令实验）。

## 实验设计

Experiment 3 测试模型是否能够通过学习"指令"（guidance）来泛化到未见过的例子：

- **训练集 (all.jsonl)**: 包含两部分
  - `guidances`: 指令格式，告诉模型"当看到问题 X 时，回答 Y"
  - `realized_examples`: 与 guidances 对应的问答对（模型在训练中会看到这些例子）

- **测试集**:
  - `realized_examples.jsonl`: 训练中见过的问答对（验证记忆）
  - `unrealized_examples.jsonl`: 未在训练中见过的问答对，但有对应的 guidances（验证泛化）

**关键假设**：如果模型真正"理解"了 guidances，应该能回答 unrealized_examples 中的问题。

## 文件结构

```
scripts/local_experiment3/
├── train_lora.py           # LoRA 微调脚本
├── eval_experiment3.py     # 评测脚本
├── aggregate_and_plot.py   # 结果聚合和绘图
├── run_exp3.sh             # 完整实验运行脚本
└── README.md               # 本文件

src/local_experiment3/
├── __init__.py
└── data.py                 # 数据处理模块
```

## 快速开始

### 1. 冒烟测试（快速验证代码）

```bash
# 训练（1 epoch，使用 Qwen3-4B）
python scripts/local_experiment3/train_lora.py \
    --model_id Qwen/Qwen3-4B \
    --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
    --num_epochs 1 \
    --max_seq_len 256 \
    --output_dir outputs/exp3_smoke_test

# 评测
python scripts/local_experiment3/eval_experiment3.py \
    --base_model_id Qwen/Qwen3-4B \
    --lora_dir outputs/exp3_smoke_test \
    --dataset_dir data/instructions/copypaste_ug100_rg1000_main \
    --out_dir outputs/exp3_smoke_test/eval \
    --max_samples 100
```

### 2. 完整实验（三个模型）

```bash
bash scripts/local_experiment3/run_exp3.sh
```

### 3. 结果聚合

```bash
python scripts/local_experiment3/aggregate_and_plot.py \
    --results_root outputs/local_exp3 \
    --out_dir outputs/local_exp3/aggregated
```

## 支持的模型

| Preset | Model ID | 说明 |
|--------|----------|------|
| qwen3_4b | Qwen/Qwen3-4B | 通义千问3 4B |
| gemma3_4b | google/gemma-3-4b-it | Gemma 3 4B Instruct |
| phi3_5_mini_instruct | microsoft/Phi-3.5-mini-instruct | Phi-3.5 Mini Instruct |

## 评测指标

- **realized_examples accuracy**: 在训练中见过的例子上的准确率（测试记忆）
- **unrealized_examples accuracy**: 在未见过的例子上的准确率（测试泛化）

根据原论文的假设，如果存在"逆转诅咒"，unrealized_examples 的准确率应该显著低于 realized_examples。

## 命令行参数

### train_lora.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model_id | - | HuggingFace 模型 ID |
| --preset | - | 预设模型（逗号分隔） |
| --dataset_dir | data/instructions/... | 数据集目录 |
| --num_epochs | 1 | 训练轮数 |
| --max_seq_len | 512 | 最大序列长度 |
| --lora_r | 16 | LoRA rank |
| --lora_alpha | 32 | LoRA alpha |
| --lr | 2e-4 | 学习率 |

### eval_experiment3.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --base_model_id | - | 基座模型 ID |
| --lora_dir | - | LoRA 权重目录 |
| --max_new_tokens | 64 | 生成最大 token 数 |
| --similarity_threshold | 0.80 | 相似度匹配阈值 |
| --batch_size | 16 | 评测批次大小 |

## 输出结构

```
outputs/local_exp3/
├── qwen3_4b/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── train_meta.json
│   └── eval/
│       ├── realized_examples.csv
│       ├── unrealized_examples.csv
│       └── summary.json
├── gemma3_4b/
│   └── ...
├── phi3_5_mini/
│   └── ...
└── aggregated/
    ├── comparison.csv
    ├── comparison.png
    └── aggregated.json
```
