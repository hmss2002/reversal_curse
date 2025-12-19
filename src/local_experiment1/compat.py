"""兼容性检查与常用小工具。

为什么需要这个文件？
- 本仓库原始依赖里 transformers 版本较老（4.28.1）。
- 你指定的模型（Llama 3.2 / Qwen3 / Gemma 3）通常需要更高版本 transformers 才能正确加载。
- 同时 LoRA 需要 peft 库。

因此我们在运行入口处做显式版本检查，避免用户在长时间下载后才报错。
"""

from __future__ import annotations

from packaging import version


def require_min_versions(*, min_transformers: str = "4.45.0", min_peft: str = "0.13.0") -> None:
    """检查运行所需的最低版本。

    参数
    - min_transformers: 允许加载新架构模型的 transformers 最低版本。
    - min_peft: LoRA/PEFT 最低版本。

    设计说明（重要）
    - 这里不自动 pip install：因为在很多训练环境里用户希望可控升级。
    - 我们只给出“明确可操作”的报错提示。
    """

    try:
        import transformers  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未安装 transformers。请先安装 requirements-local-models.txt 里的依赖。"
        ) from e

    try:
        import peft  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未安装 peft（LoRA 必需）。请先安装 requirements-local-models.txt 里的依赖。"
        ) from e

    if version.parse(transformers.__version__) < version.parse(min_transformers):
        raise RuntimeError(
            "当前 transformers 版本过低："
            f"{transformers.__version__} < {min_transformers}。\n"
            "这会导致 Llama-3.2 / Qwen3 / Gemma 3 等模型无法正确 from_pretrained。\n"
            "请执行：pip install -r requirements-local-models.txt（建议在 conda 环境里）。"
        )

    if version.parse(peft.__version__) < version.parse(min_peft):
        raise RuntimeError(
            "当前 peft 版本过低："
            f"{peft.__version__} < {min_peft}。\n"
            "请执行：pip install -r requirements-local-models.txt。"
        )
