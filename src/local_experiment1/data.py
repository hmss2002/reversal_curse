"""Experiment 1（Reversing identities）本地训练/评测所需的数据处理。

核心点：
1) 原始数据是 JSONL，每行包含：
   - prompt: 作为“输入前缀”
   - completion: 作为“希望模型生成的后缀”

2) 对因果语言模型（CausalLM）微调时，我们通常把 prompt+completion 拼成一段 text。
   - input_ids = tokenize(prompt + completion)
   - labels 与 input_ids 同形状
   - 但 labels 中 prompt 部分必须 mask 掉（置为 -100），否则模型会被训练去复述 prompt。

3) 我们不靠 epoch 对齐训练口径，而是靠 token_budget：
   - token_budget 的统计方式在训练脚本里做（按 attention_mask.sum()）。
   - 数据侧只要能稳定产出 input_ids/labels 即可。

本文件包含：
- JSONL 读取
- 单样本编码（input_ids/labels）
- DataCollator（动态 padding，labels 用 -100 padding）

注意：这里写了很多中文注释，目的是让你后续改动更省心。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。

    参数
    - path: JSONL 文件路径

    返回
    - List[dict]，每个 dict 至少包含 prompt/completion 字段
    """

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class PromptCompletionDataset(Dataset):
    """把 prompt/completion JSONL 转成可训练样本。

    这里使用 torch.utils.data.Dataset（非 IterableDataset），
    适合数据量不大（Experiment 1 只有几千行）。

    关键输出字段：
    - input_ids: [seq]
    - attention_mask: [seq]
    - labels: [seq]，prompt 部分为 -100

    备注：
    - 我们不在这里做 padding；padding 交给 DataCollator。
    - max_seq_len 做截断，保证显存可控。
    """

    def __init__(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
        drop_if_no_target_tokens: bool = True,
    ) -> None:
        self._rows = list(rows)
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._drop_if_no_target_tokens = drop_if_no_target_tokens

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._rows[idx]
        prompt: str = row["prompt"]
        completion: str = row["completion"]

        # 1) 先对 prompt 单独 tokenize，用来定位 prompt 的 token 长度。
        #    注意：prompt/completion 在原始数据里通常通过“completion 前导空格”来保证自然衔接。
        prompt_enc = self._tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_seq_len,
        )
        prompt_ids: List[int] = prompt_enc["input_ids"]

        # 2) 再对 prompt+completion 整体 tokenize，得到训练时模型真实看到的 token 序列。
        full_text = prompt + completion
        full_enc = self._tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_seq_len,
        )
        input_ids: List[int] = full_enc["input_ids"]

        # 3) 计算 prompt 在 full 序列中的“前缀长度”。
        #    理想情况：prompt_ids 应当是 input_ids 的前缀。
        #    但少数 tokenizer 对边界空格的处理可能导致不完全一致。
        prompt_len = _safe_prefix_length(prompt_ids, input_ids)

        # 4) labels：默认等于 input_ids，但 prompt 部分全部 mask 掉（-100）。
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # 5) 如果截断导致 completion 没有剩余 token，训练这个样本就没有意义；
        #    可选择丢掉（drop_if_no_target_tokens=True）。
        if self._drop_if_no_target_tokens and all(x == -100 for x in labels):
            # 这里返回一个“空样本”的约定：由 collator/loader 负责跳过。
            # （比 raise 更稳：不会因为个别异常样本中断整个训练。）
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long),
            }

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _safe_prefix_length(prefix: List[int], full: List[int]) -> int:
    """计算 prefix 在 full 中作为“前缀”匹配的最大长度。

    为什么不用 len(prefix)？
    - 有些 tokenizer 在字符串拼接边界处会发生 token 合并/切分差异；
      导致 prefix_ids 不再严格等于 full_ids[:len(prefix_ids)]。

    我们的策略：
    - 取二者的最长公共前缀长度（LCP）。
    - 这样 labels mask 至少不会越界。

    这不是最完美的对齐，但对 prompt+completion 的训练来说足够稳健。
    """

    n = min(len(prefix), len(full))
    i = 0
    while i < n and prefix[i] == full[i]:
        i += 1
    return i


@dataclass
class DataCollatorForCausalLMPromptCompletion:
    """把一批样本 pad 成同长度。

    关键点：
    - input_ids 用 pad_token_id padding
    - attention_mask 对 padding 位置为 0
    - labels 对 padding 位置为 -100（保持 loss ignore）

    另外，我们会跳过“空样本”（长度为 0），
    这是 Dataset 在截断后无 target token 时返回的。
    """

    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 过滤空样本
        features = [f for f in features if f["input_ids"].numel() > 0]
        if not features:
            # 理论上不该发生（除非 max_seq_len 极小），这里兜底。
            return {
                "input_ids": torch.zeros((0, 0), dtype=torch.long),
                "attention_mask": torch.zeros((0, 0), dtype=torch.long),
                "labels": torch.zeros((0, 0), dtype=torch.long),
            }

        max_len = max(int(f["input_ids"].numel()) for f in features)

        def pad_1d(x: torch.Tensor, *, pad_value: int) -> torch.Tensor:
            if x.numel() == max_len:
                return x
            pad = torch.full((max_len - x.numel(),), pad_value, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(f["input_ids"], pad_value=self.pad_token_id) for f in features])
        attention_mask = torch.stack([pad_1d(f["attention_mask"], pad_value=0) for f in features])
        labels = torch.stack([pad_1d(f["labels"], pad_value=-100) for f in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
