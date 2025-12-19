"""LoRA/PEFT 相关封装。

你要求“统一训练策略：建议统一用 LoRA”。
因此我们把：
- LoRA 配置（r/alpha/dropout）
- target_modules 的默认选择（不同模型结构略有差异）
集中在这里，避免训练脚本里写一堆 if/else。

说明：
- target_modules 并不存在唯一正确答案；
- 这里提供一个“业界常用且相对安全”的默认值；
- 同时允许 CLI 覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


@dataclass
class LoRAArgs:
    # LoRA 的秩（rank）。越大可训练参数越多，也越吃显存。
    r: int = 16
    # LoRA 的缩放系数。常见设置是 2r 或 4r。
    lora_alpha: int = 32
    # Dropout，用于抑制过拟合。
    lora_dropout: float = 0.05


def default_lora_target_modules(model: Any) -> List[str]:
    """给出默认 `target_modules`。

    你希望“统一训练策略：LoRA”，但不同开源模型的线性层命名并不一致。
    如果 target_modules 选错，PEFT 会直接报错（找不到模块名）。

    这里采用一个“尽量少出错”的策略：
    1) 先扫描模型的 `named_modules()`，根据实际出现的模块后缀自动判断属于哪一类结构。
    2) 常见结构：
       - LLaMA/Gemma/Qwen 等：q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/down_proj
       - GPT-2 系：c_attn/c_proj（以及可选的 c_fc）
    3) 如果仍无法判断，则退回 LLaMA-like 的默认值，并把问题留给 CLI 覆盖。

    说明：
    - 对你指定的三大模型（Llama-3.2 / Qwen3 / Gemma 3）通常会落在 LLaMA-like 分支。
    - GPT-2 分支主要用于我们在开发阶段做 tiny 模型冒烟测试。
    """

    module_suffixes = _collect_module_suffixes(model.named_modules())

    # 1) LLaMA-like：你的主目标模型基本都在这里
    if {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(module_suffixes):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # 2) GPT-2：用于 tiny-gpt2 等快速冒烟
    # GPT-2 的注意力通常是：attn.c_attn / attn.c_proj
    # MLP 常见：mlp.c_fc / mlp.c_proj（注意这里 c_proj 会同时命中 attn 与 mlp）
    if "c_attn" in module_suffixes:
        targets = ["c_attn", "c_proj"]
        if "c_fc" in module_suffixes:
            targets.append("c_fc")
        return targets

    # 3) 兜底：保持与之前版本一致
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def _collect_module_suffixes(named_modules: Iterable[tuple[str, Any]]) -> set[str]:
    """收集 `named_modules()` 中每个模块名的“最后一段后缀”。

    例：
    - transformer.h.0.attn.c_attn -> c_attn
    - model.layers.0.self_attn.q_proj -> q_proj

    这样我们可以用极低成本判断当前模型更像哪种结构。
    """

    suffixes: set[str] = set()
    for name, _module in named_modules:
        if not name:
            continue
        suffixes.add(name.rsplit(".", 1)[-1])
    return suffixes


def apply_lora(
    *,
    model: Any,
    lora: LoRAArgs,
    target_modules: Optional[List[str]] = None,
) -> Any:
    """把 LoRA 适配器挂到模型上，并返回 PEFT 包装后的模型。

    重要：
    - PEFT 会把原模型参数默认冻结，仅训练 LoRA 参数（满足你“统一 LoRA 策略”的要求）。
    - 返回对象通常是 peft.PeftModel 或其子类。
    """

    from peft import LoraConfig, get_peft_model  # type: ignore

    if target_modules is None:
        target_modules = default_lora_target_modules(model)

    # task_type 选择 CAUSAL_LM，适用于 decoder-only LM
    config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    # 打印可训练参数比例（非常有助于确认确实在做 LoRA，而不是全参）
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model
