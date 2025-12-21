"""Experiment 3（Reversing instructions）本地训练/评测所需的数据处理。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。"""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class Exp3Dataset(Dataset):
    """把 Experiment 3 的 prompt/completion JSONL 转成可训练样本。"""

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
        prompt: str = row.get("prompt", "")
        completion: str = row["completion"]

        # 1) 先对 prompt 单独 tokenize
        if prompt:
            prompt_enc = self._tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self._max_seq_len,
            )
            prompt_ids: List[int] = prompt_enc["input_ids"]
        else:
            prompt_ids = []

        # 2) 对 prompt+completion 整体 tokenize
        full_text = prompt + completion
        full_enc = self._tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_seq_len,
        )
        input_ids: List[int] = full_enc["input_ids"]

        # 3) 计算 prompt 在 full 序列中的前缀长度
        prompt_len = _safe_prefix_length(prompt_ids, input_ids)

        # 4) labels：prompt 部分 mask 掉
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # 5) 如果截断导致 completion 没有剩余 token
        if self._drop_if_no_target_tokens and all(x == -100 for x in labels):
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
    """计算 prefix 在 full 中作为前缀匹配的最大长度。"""
    n = min(len(prefix), len(full))
    i = 0
    while i < n and prefix[i] == full[i]:
        i += 1
    return i


@dataclass
class DataCollatorForExp3:
    """把一批样本 pad 成同长度。"""

    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 过滤空样本
        features = [f for f in features if f["input_ids"].numel() > 0]
        if not features:
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
