"""知途 · 环境自检（迁移后第一件事就跑这个）。

用法：
    cd Qin-path
    python scripts/check_env.py            # 只检查，人类可读
    python scripts/check_env.py --quiet     # 只在有问题时输出（给脚本调用）

设计原则：
- **不依赖任何第三方库**（只用标准库 + 项目自身的 ml.config），
  这样在依赖还没装齐的机器上也能跑起来，它能告诉你缺什么。
- 只做只读检查，不改任何文件、不建目录、不训练。
- 退出码：0 = 没有致命问题；1 = 有缺失项（依赖导入失败 / 不在项目根目录）。

为什么需要它：这个项目要求「一份代码两种设备」，迁移到有显卡的机器后，
最要紧的是确认 CUDA 到底有没有生效、当前实际跑的是哪个档位。
看错档位会导致拿 CPU 档的指标去当 GPU 档结论用。
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# 直接 `python scripts/check_env.py` 时 sys.path[0] 是 scripts/ 而不是项目根目录，
# 会导致 `import ml.config` 失败。这里显式把项目根目录挂上去，
# 这样从任何工作目录调用本脚本都不会因为路径问题误报"依赖缺失"。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OK, WARN, FAIL = "OK", "警告", "缺失"
WIDTH = 30


def _row(tag: str, title: str, detail: str = "") -> tuple[str, str, str]:
    return tag, title, detail


def _module_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"{type(exc).__name__}: {exc}") from exc
    return str(getattr(mod, "__version__", "?"))


def collect() -> tuple[list[tuple[str, str, str]], dict]:
    rows: list[tuple[str, str, str]] = []
    facts: dict = {}

    # ---------------------------------------------------------------- 位置
    root = PROJECT_ROOT
    in_root = (root / "ml" / "config.py").is_file()
    rows.append(_row(OK if in_root else FAIL, "项目根目录",
                     str(root) if in_root else f"{root} —— 这里找不到 ml/config.py"))
    cwd = Path.cwd()
    # 后面的 `python -m ml.xxx` 都要求工作目录是项目根，所以这里单独提一句
    rows.append(_row(
        OK if cwd == root else WARN, "当前工作目录",
        str(cwd) if cwd == root else f"{cwd} —— 不是项目根，`python -m ml.xxx` 会报 No module named ml",
    ))

    # ---------------------------------------------------------------- Python
    vi = sys.version_info
    rows.append(_row(OK if vi >= (3, 10) else FAIL, "Python 版本",
                     f"{platform.python_version()}（要求 >= 3.10）"))
    in_venv = sys.prefix != sys.base_prefix
    rows.append(_row(OK, "虚拟环境",
                     sys.prefix if in_venv else "未启用 venv（当前是系统 Python，建议建一个）"))

    # ---------------------------------------------------------------- 依赖
    # 必修：跑实验和后端都离不开
    required = ["torch", "numpy", "sklearn", "fastapi", "sqlalchemy", "bcrypt",
                "uvicorn", "pytest"]
    # 选修：缺了不影响主流程
    optional = [("websocket", "scripts/shot.py 无头截图需要"),
                ("pandas", "可选，本项目未强依赖"),
                ("httpx", "fastapi TestClient 需要")]
    missing: list[str] = []
    for name in required:
        try:
            rows.append(_row(OK, f"依赖 {name}", _module_version(name)))
        except ImportError as exc:
            missing.append(name)
            rows.append(_row(FAIL, f"依赖 {name}", f"导入失败 {exc}"))
    for name, why in optional:
        try:
            rows.append(_row(OK, f"可选 {name}", _module_version(name)))
        except ImportError:
            rows.append(_row(WARN, f"可选 {name}", f"未安装（{why}）"))
    facts["missing_required"] = missing

    # ---------------------------------------------------------------- 算力
    cuda_ok = False
    try:
        import torch  # noqa: PLC0415

        cuda_ok = bool(torch.cuda.is_available())
        rows.append(_row(OK if cuda_ok else WARN, "torch.cuda.is_available()", str(cuda_ok)))
        rows.append(_row(OK, "torch 编译的 CUDA 版本",
                         str(torch.version.cuda) if torch.version.cuda else "None ← 这是 CPU 版 torch"))
        if cuda_ok:
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                rows.append(_row(
                    OK, f"GPU {i}",
                    f"{prop.name}｜算力 {prop.major}.{prop.minor}｜"
                    f"显存 {prop.total_memory / 1024 ** 3:.1f} GB",
                ))
    except ImportError:
        rows.append(_row(FAIL, "torch", "导入失败，无法判断算力"))
    facts["cuda_ok"] = cuda_ok

    # ---------------------------------------------------------------- 档位
    if in_root:
        try:
            from ml.config import DEVICE, get_profile  # noqa: PLC0415

            profile = get_profile()
            facts["device"] = DEVICE
            facts["profile"] = profile
            # 探测结果与实际档位必须自洽，否则说明有人手动设了 ZT_DEVICE / ZT_PROFILE
            rows.append(_row(OK if (DEVICE == "cuda") == cuda_ok else WARN,
                             "resolve_device() 探测结果", DEVICE))
            consistent = (DEVICE == "cuda") == (profile.name == "gpu")
            rows.append(_row(
                OK if consistent else WARN, "实际使用档位",
                f"{profile.name}｜max_seq_len={profile.max_seq_len} epochs={profile.epochs} "
                f"batch={profile.batch_size} d_model={profile.d_model}",
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(FAIL, "ml.config", f"导入失败：{type(exc).__name__}: {exc}"))

    # ---------------------------------------------------------------- 数据
    csv_path = root / "data" / "raw" / "skill_builder_data_corrected.csv"
    if csv_path.is_file():
        rows.append(_row(OK, "真实数据集 CSV",
                         f"{csv_path.name}（{csv_path.stat().st_size / 1e6:.1f} MB）"))
    else:
        rows.append(_row(WARN, "真实数据集 CSV",
                         "缺失 → 需重新下载（见 BUDDYREADME 第 2 步）"))
    facts["has_csv"] = csv_path.is_file()

    for fname, label, why in (
        ("dataset_index.json", "切分（合成 / 演示）", "后端启动依赖它，缺了后端会起不来"),
        ("assistments2009_index.json", "切分（真实 / 答辩）", "答辩指标依赖它"),
    ):
        f = root / "data" / "processed" / fname
        rows.append(_row(OK if f.is_file() else WARN, label,
                         "已就绪" if f.is_file() else f"{fname} 缺失 → {why}，需重建"))
    facts["has_synth_split"] = (root / "data" / "processed" / "dataset_index.json").is_file()
    facts["has_real_split"] = (root / "data" / "processed" / "assistments2009_index.json").is_file()

    # ---------------------------------------------------------------- 产物
    results = sorted((root / "ml" / "artifacts").glob("results_*.json"))
    rows.append(_row(OK if results else WARN, "实验指标文件",
                     "、".join(r.name for r in results) if results else "无 → 还没跑过实验"))
    models = sorted((root / "ml" / "artifacts" / "models").glob("*"))
    rows.append(_row(OK if models else WARN, "模型产物",
                     f"{len(models)} 个文件" if models else "无"))

    # ---------------------------------------------------------------- 其它
    db = root / "backend" / "zhitu.db"
    rows.append(_row(OK if db.is_file() else WARN, "演示数据库",
                     "已存在" if db.is_file() else "缺失 → python -m backend.app.seed"))
    facts["has_db"] = db.is_file()

    node = shutil.which("node")
    if node:
        try:
            ver = subprocess.run([node, "--version"], capture_output=True, text=True,
                                 timeout=15).stdout.strip()
        except Exception:  # noqa: BLE001
            ver = "?"
        rows.append(_row(OK, "Node.js", ver))
    else:
        rows.append(_row(WARN, "Node.js", "未找到（只跑后端不影响，前端要它）"))
    nm = root / "frontend" / "node_modules"
    rows.append(_row(OK if nm.is_dir() else WARN, "前端依赖",
                     "node_modules 已存在" if nm.is_dir() else "缺失 → cd frontend && npm install"))

    return rows, facts


def verdict(rows, facts) -> str:
    if not (PROJECT_ROOT / "ml" / "config.py").is_file():
        return ("【结论】找不到项目代码 —— 本脚本的位置决定了项目根目录，"
                "它应该在 Qin-path/scripts/ 下。请确认代码是完整克隆/解压出来的。")
    if facts.get("missing_required"):
        return ("【结论】环境不完整：必修依赖导不进来，先装依赖再谈别的。\n"
                "         pip install -r requirements.txt\n"
                "         （有显卡的机器改用 requirements-gpu.txt 的步骤：先装 CUDA 版 torch）")
    device = facts.get("device")
    profile = facts.get("profile")
    if facts.get("cuda_ok") and profile is not None and profile.name == "gpu":
        return ("【结论】环境就绪，且已自动切到 gpu 档。\n"
                "         下一步按 BUDDYREADME 的「迁移三步」：重下数据 → 重建切分(--max-seq-len 200)\n"
                "         → 跑实验(--profile gpu) → 更新文档里的指标数字。")
    if facts.get("cuda_ok"):
        return ("【结论】CUDA 可用，但当前档位仍是 cpu（可能被 ZT_PROFILE / --profile 固定成 cpu 了）。\n"
                "         确认后再跑实验：set ZT_PROFILE=gpu（或每条命令显式加 --profile gpu）。")
    return ("【结论】没有可用的 CUDA。如果这台机器确实有显卡，按顺序排查：\n"
            "         1) 装的 torch 是不是 CUDA 版 —— 看上面「torch 编译的 CUDA 版本」是不是 None；\n"
            "            是 None 就说明装成了 CPU 版，重装：\n"
            "            pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "         2) 显卡驱动是否正常（nvidia-smi 能不能列出显卡）；\n"
            "         3) 驱动版本与 torch 要求的 CUDA 运行时是否匹配。\n"
            "         没解决之前不要跑实验 —— CPU 档的数字不能当 GPU 档结论用。")


def main() -> int:
    ap = argparse.ArgumentParser(description="知途环境自检")
    ap.add_argument("--quiet", action="store_true", help="只在有问题时输出")
    args = ap.parse_args()

    rows, facts = collect()
    text = verdict(rows, facts)
    fail_n = sum(1 for t, _, _ in rows if t == FAIL)
    warn_n = sum(1 for t, _, _ in rows if t == WARN)

    if not args.quiet or fail_n or warn_n:
        line = "-" * 72
        print("=" * 72)
        print("知途 · 环境自检")
        print("=" * 72)
        print(f"{'状态':<6}{'检查项':<{WIDTH}}说明")
        print(line)
        for tag, title, detail in rows:
            print(f"{tag:<6}{title:<{WIDTH}}{detail}")
        print(line)
        print(f"合计 {len(rows)} 项：缺失 {fail_n}，警告 {warn_n}")
        print()
        print(text)
    return 1 if fail_n else 0


if __name__ == "__main__":
    sys.exit(main())
