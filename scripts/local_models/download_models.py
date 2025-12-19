"""可选：预下载 HuggingFace 开源模型到本地缓存。

你说“我要你自己下载开源模型”。
严格来说：
- `AutoModelForCausalLM.from_pretrained(model_id)` 本身就会自动下载并缓存；
- 但很多时候我们希望“先把模型权重下完”，避免训练中途因为网络/权限失败。

本脚本使用 huggingface_hub.snapshot_download：
- 支持指定 `--local_dir` 把模型放到固定目录
- 支持 `--token`（对需要同意协议的模型，如 Llama/Gemma）

用法：
python scripts/local_models/download_models.py --preset llama3_2_3b,qwen3_4b,gemma3_4b --local_dir models/hf

提示：
- 如果你在国内网络环境，可能需要配置 HF 镜像或代理。
- Llama/Gemma 往往需要你在 HuggingFace 上同意 license，并提供 token。
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from huggingface_hub import snapshot_download

try:
    # huggingface_hub 目前把 gated 错误类型放在内部模块里
    from huggingface_hub.utils._errors import GatedRepoError  # type: ignore
except Exception:  # pragma: no cover
    GatedRepoError = None  # type: ignore


MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "llama3_2_3b": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    },
    "qwen3_4b": {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    },
    "gemma3_4b": {
        "repo_id": "google/gemma-3-4b-it",
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
    },

    # Llama 无法获批时的替代模型（公开可下载）
    "qwen2_5_3b_instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    },
    "phi3_5_mini_instruct": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
    },
    "falcon3_3b_instruct": {
        "repo_id": "tiiuae/falcon3-3b-instruct",
        "revision": "411bb94318f94f7a5735b77109f456b1e74b42a1",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preset",
        type=str,
        required=True,
        help=(
            "Comma-separated presets. Common ones: "
            "qwen3_4b, gemma3_4b, phi3_5_mini_instruct, qwen2_5_3b_instruct, falcon3_3b_instruct "
            "(llama3_2_3b may be gated/denied)."
        ),
    )
    p.add_argument("--local_dir", type=str, default="models/hf", help="Where to store downloaded snapshots")
    p.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token (or set env HF_TOKEN)",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="",
        help="Model revision override (branch/tag/commit). Empty => use preset pinned sha.",
    )
    p.add_argument(
        "--verify_only",
        action="store_true",
        help="只下载少量元数据文件，用来验证权限/网络；不会下载大权重。",
    )
    p.add_argument(
        "--allow_patterns",
        type=str,
        default="",
        help="自定义允许下载的文件通配（逗号分隔）。为空则由 --verify_only 决定。",
    )
    p.add_argument(
        "--no_token",
        action="store_true",
        help=(
            "显式禁用鉴权 token（即使环境变量 HF_TOKEN 已设置也不会使用）。"
            "对公开模型（如 Qwen）很有用，可避免环境变量里的无效 token 影响下载。"
        ),
    )
    return p.parse_args()


def _validate_token_or_raise(token: str) -> None:
    """确保 token 适合放进 HTTP header。

    为什么需要它：
    - requests/urllib3 在写 header 时会用 latin-1 编码。
    - 如果你把 HF_TOKEN 设成中文占位符（例如“你的huggingface_token”），
      就会出现你现在看到的 UnicodeEncodeError。
    """

    if not token:
        return

    # HuggingFace token 本质上是 ASCII 字符串。
    try:
        token.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(
            "HF token 必须是 ASCII 字符串；看起来你传入了包含中文/非 ASCII 的占位符。"
            "\n请把环境变量 HF_TOKEN 设置为真实的 HuggingFace token（例如 hf_...），"
            "或对公开模型使用 --no_token 来忽略 token。"
        ) from e


def main() -> None:
    args = parse_args()

    presets = [x.strip() for x in args.preset.split(",") if x.strip()]
    os.makedirs(args.local_dir, exist_ok=True)

    # token 选择策略：
    # - 若 --no_token：强制禁用（huggingface_hub 支持 token=False）
    # - 否则优先用 --token；未提供时让 huggingface_hub 自行从环境/本地缓存取
    #   （但如果环境变量里是中文占位符，这会导致 requests header 编码崩溃，所以我们做校验）
    if args.no_token:
        resolved_token = False  # type: ignore[assignment]
    else:
        if args.token:
            _validate_token_or_raise(args.token)
            resolved_token = args.token
        else:
            # 未显式提供 token 时，如果环境里有 HF_TOKEN，最好也校验一下，避免“中文占位符”污染。
            env_token = os.environ.get("HF_TOKEN", "")
            if env_token:
                _validate_token_or_raise(env_token)
            resolved_token = None

    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"[env] HF_ENDPOINT={endpoint}")

    for key in presets:
        if key not in MODEL_PRESETS:
            raise ValueError(f"Unknown preset: {key}")

        repo_id = MODEL_PRESETS[key]["repo_id"]
        revision = args.revision or MODEL_PRESETS[key]["revision"]
        print(f"[download] {key} => {repo_id}")

        # allow_patterns 的设计目的：
        # - 你想“预下载验证权限”，不想一上来就拉几十 GB 权重。
        # - gated 模型（Llama/Gemma）如果没权限，会在这里直接报 401/403。
        allow_patterns = None
        if args.allow_patterns.strip():
            allow_patterns = [x.strip() for x in args.allow_patterns.split(",") if x.strip()]
        elif args.verify_only:
            allow_patterns = [
                "config.json",
                "tokenizer*",
                "special_tokens_map.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "*.json",
            ]

        try:
            path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=os.path.join(args.local_dir, repo_id.split("/")[-1]),
                local_dir_use_symlinks=False,
                token=resolved_token,
                allow_patterns=allow_patterns,
            )
            print(f"[done] {repo_id} (revision={revision}) saved to: {path}")
        except Exception as e:
            # 403 gated：给出更可操作的指引
            is_gated = (GatedRepoError is not None) and isinstance(e, GatedRepoError)
            if is_gated or ("GatedRepoError" in type(e).__name__):
                print("[error] gated repo: access forbidden (403).")
                print(f"        repo: {repo_id}")
                print("        下一步：")
                print(f"        1) 打开 https://huggingface.co/{repo_id} 并点击 Request/Agree（同意协议/申请访问权限）")
                print("        2) 确认你使用的 token 属于已获授权的账号（HF_TOKEN 或 --token）")
                if endpoint and "hf-mirror" in endpoint:
                    print("        3) 你当前在用 hf-mirror（HF_ENDPOINT 指向镜像）。gated 模型常在镜像上无法鉴权。")
                    print("           建议临时切回官方：export HF_ENDPOINT=https://huggingface.co")
                raise

            # 其他错误原样抛出（保留堆栈，方便你定位网络/磁盘等问题）
            raise


if __name__ == "__main__":
    main()
