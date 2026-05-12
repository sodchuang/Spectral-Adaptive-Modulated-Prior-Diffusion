"""
Robomimic 資料集下載 + 絕對動作轉換腳本
用途：夾爪夾取任務 (lift) 的 proficient human 示範資料

使用方式：
  python download_data.py              # 下載 + 轉換（預設 lift ph）
  python download_data.py --skip-download  # 僅轉換（已有原始檔時）
  python download_data.py --task can   # 換成其他任務

資料路徑：
  原始：  data/robomimic/datasets/lift/ph/low_dim.hdf5
  轉換後：data/robomimic/datasets/lift/ph/low_dim_abs.hdf5  ← config 使用此路徑
"""

import os
import sys
import pathlib
import argparse
import subprocess

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data" / "robomimic" / "datasets"

# ── 可下載的任務清單 ────────────────────────────────────────────────────────
#   只列出「低維度觀測 + proficient human」示範，夾爪任務優先
TASKS = {
    "lift": {
        "desc": "夾爪夾取方塊（最簡單，推薦首選）",
        "url": (
            "http://downloads.cs.stanford.edu/downloads/rt_benchmark/"
            "lift/ph/low_dim_v141.hdf5"
        ),
        "raw_name": "low_dim.hdf5",
    },
    "can": {
        "desc": "夾爪撿起鋁罐放入桶中",
        "url": (
            "http://downloads.cs.stanford.edu/downloads/rt_benchmark/"
            "can/ph/low_dim_v141.hdf5"
        ),
        "raw_name": "low_dim.hdf5",
    },
    "square": {
        "desc": "夾爪將螺帽套入螺柱（需精確對齊）",
        "url": (
            "http://downloads.cs.stanford.edu/downloads/rt_benchmark/"
            "square/ph/low_dim_v141.hdf5"
        ),
        "raw_name": "low_dim.hdf5",
    },
}


def download(task: str):
    info = TASKS[task]
    out_dir = DATA_DIR / task / "ph"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / info["raw_name"]

    if dest.exists():
        print(f"[skip] 原始檔已存在：{dest}")
        return dest

    print(f"[下載] {task}  →  {dest}")
    print(f"       URL: {info['url']}")

    try:
        # 優先用 robomimic 官方下載工具（較穩定）
        subprocess.run(
            [
                sys.executable, "-m",
                "robomimic.scripts.download_datasets",
                "--tasks", task,
                "--dataset_types", "ph",
                "--hdf5_types", "low_dim",
                "--download_dir", str(DATA_DIR),
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # fallback：直接 wget / curl
        print("[warn] robomimic 下載工具失敗，改用 wget/curl …")
        if _which("wget"):
            subprocess.run(["wget", "-O", str(dest), info["url"]], check=True)
        elif _which("curl"):
            subprocess.run(["curl", "-L", "-o", str(dest), info["url"]], check=True)
        else:
            print("[error] 找不到 wget / curl，請手動下載：")
            print(f"        {info['url']}")
            print(f"        → 存至 {dest}")
            sys.exit(1)

    return dest


def convert(task: str, raw_path: pathlib.Path, num_workers: int):
    out_path = raw_path.parent / "low_dim_abs.hdf5"

    if out_path.exists():
        print(f"[skip] 轉換檔已存在：{out_path}")
        return out_path

    print(f"[轉換] 絕對動作轉換中（workers={num_workers}）…")
    script = ROOT / "diffusion_policy" / "scripts" / "robomimic_dataset_conversion.py"
    subprocess.run(
        [
            sys.executable, str(script),
            "-i", str(raw_path),
            "-o", str(out_path),
            "-n", str(num_workers),
        ],
        check=True,
    )
    print(f"[完成] 轉換後檔案：{out_path}")
    return out_path


def _which(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None


def main():
    parser = argparse.ArgumentParser(description="下載並轉換 robomimic 夾爪任務資料集")
    parser.add_argument(
        "--task", default="lift",
        choices=list(TASKS.keys()),
        help="任務名稱（預設 lift）",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="跳過下載，只執行轉換（原始檔須已存在）",
    )
    parser.add_argument(
        "--num-workers", type=int, default=os.cpu_count(),
        help="轉換時使用的 CPU 核心數（預設全核）",
    )
    args = parser.parse_args()

    print(f"\n=== SAMP_Diff_v1 資料集準備 ===")
    print(f"任務：{args.task}  —  {TASKS[args.task]['desc']}")
    print(f"目標：{DATA_DIR / args.task / 'ph' / 'low_dim_abs.hdf5'}\n")

    raw_path = DATA_DIR / args.task / "ph" / "low_dim.hdf5"

    if not args.skip_download:
        raw_path = download(args.task)
    else:
        if not raw_path.exists():
            print(f"[error] 原始檔不存在：{raw_path}")
            print("        請先執行不帶 --skip-download 的指令下載。")
            sys.exit(1)

    abs_path = convert(args.task, raw_path, args.num_workers)

    print(f"\n✓ 完成！訓練指令：")
    print(f"  python train.py --config-name={args.task}_ph\n")


if __name__ == "__main__":
    main()
