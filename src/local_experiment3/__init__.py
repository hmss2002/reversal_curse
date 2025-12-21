"""Local Experiment 3: Reversing Instructions

这是对原始 Experiment 3 的本地（开源模型）复现。

实验设计：
- 训练集：all.jsonl (包含 guidances + realized_examples)
  - guidances: 指令格式，告诉模型如何回答特定问题
  - realized_examples: 与 guidances 对应的问答对（模型会在训练中看到）
- 测试集：unrealized_examples.jsonl
  - 模型从未在训练中看到的问答对，但有对应的 guidances
  - 如果模型真正"理解"了 guidances，应该能回答这些问题

关键评测指标：
- realized accuracy: 模型在训练中见过的例子上的准确率（应该很高）
- unrealized accuracy: 模型在未见过的例子上的准确率（测试泛化能力）
"""
