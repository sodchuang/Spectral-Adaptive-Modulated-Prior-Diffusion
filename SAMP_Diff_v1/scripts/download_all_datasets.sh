#!/usr/bin/env bash
# =============================================================================
#  download_all_datasets.sh
#  SAMP_Diff_v1 — 全資料集下載腳本
#
#  夾爪（Parallel Gripper）相容性說明：
#    ✓  Robomimic   lift / can / square / transport   (純夾爪任務)
#    ✓  PushT       2-D 推塊                         (無手臂，純 2D)
#    ✓  MimicGen    lift_d0 / can_d0 / square_d0      (夾爪擴增)
#    ✓  Bridge V2   廚房物件抓放                     (夾爪末端執行器)
#    ✓  DROID       各種夾爪抓取                     (夾爪末端執行器)
#    ✓  ManiSkill2  PickCube / StackCube 等           (平行夾爪)
#    ✓  Open X      篩選 parallel_gripper 末端         (夾爪子集)
#    ✗  Adroit      24-DOF 靈巧手                    (跳過)
#    -  Meta-World  simulation library，無需下載資料集
#    -  UR_Real_Data 自行錄製，腳本不下載
#
#  用法：
#    chmod +x scripts/download_all_datasets.sh
#    ./scripts/download_all_datasets.sh              # 標準資料集
#    ./scripts/download_all_datasets.sh --with-bridge  # 加 Bridge V2 (~10 TB)
#    ./scripts/download_all_datasets.sh --with-droid   # 加 DROID (~15 TB)
#    ./scripts/download_all_datasets.sh --with-openx   # 加 Open X (超大)
#    ./scripts/download_all_datasets.sh --all-large    # 以上全部
#
#  前置需求：
#    conda activate robodiff
#    pip install robomimic mani-skill2 huggingface_hub tensorflow-datasets
# =============================================================================

set -euo pipefail

# ─── 路徑設定 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"          # SAMP_Diff_v1/
DATA_ROOT="$ROOT_DIR/data"

# ─── 旗標解析 ────────────────────────────────────────────────────────────────
WITH_BRIDGE=false
WITH_DROID=false
WITH_OPENX=false

for arg in "$@"; do
    case $arg in
        --with-bridge)  WITH_BRIDGE=true ;;
        --with-droid)   WITH_DROID=true  ;;
        --with-openx)   WITH_OPENX=true  ;;
        --all-large)    WITH_BRIDGE=true; WITH_DROID=true; WITH_OPENX=true ;;
        *) echo "[warn] 未知參數：$arg"; exit 1 ;;
    esac
done

# ─── 工具函式 ────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
skip()  { echo -e "\033[0;90m[SKIP]\033[0m  $*（已存在）"; }
header(){ echo -e "\n\033[1;36m════ $* ════\033[0m"; }

require_cmd() {
    if ! command -v "$1" &>/dev/null; then
        warn "找不到 $1，請先安裝：$2"; return 1
    fi
}

download_file() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then skip "$dest"; return; fi
    mkdir -p "$(dirname "$dest")"
    info "下載 $(basename "$dest") …"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    else
        curl -L --progress-bar -o "$dest" "$url"
    fi
    ok "$dest"
}

# ─── 轉換 robomimic raw → abs ────────────────────────────────────────────────
convert_robomimic() {
    local task="$1" raw="$2"
    local abs="${raw%low_dim.hdf5}low_dim_abs.hdf5"
    if [[ -f "$abs" ]]; then skip "$abs"; return; fi
    info "轉換絕對動作座標：$task …"
    python "$ROOT_DIR/diffusion_policy/scripts/robomimic_dataset_conversion.py" \
        -i "$raw" -o "$abs" -n "$(nproc)"
    ok "$abs"
}

# =============================================================================
# 1. ROBOMIMIC — 夾爪任務 (lift / can / square / transport)
# =============================================================================
header "Robomimic (MuJoCo + 夾爪)"

ROBO_BASE="$DATA_ROOT/robomimic/datasets"

# 官方 downloader（優先）
if python -c "import robomimic" &>/dev/null 2>&1; then
    info "使用 robomimic 官方下載工具 …"
    python -m robomimic.scripts.download_datasets \
        --tasks lift can square transport \
        --dataset_types ph \
        --hdf5_types low_dim \
        --download_dir "$ROBO_BASE"
    ok "Robomimic ph/low_dim 下載完成"

    # multi-human 版本（更多示範）
    python -m robomimic.scripts.download_datasets \
        --tasks lift can square \
        --dataset_types mh \
        --hdf5_types low_dim \
        --download_dir "$ROBO_BASE"
    ok "Robomimic mh/low_dim 下載完成"
else
    warn "robomimic 未安裝，改用直接下載 …"
    BASE_URL="http://downloads.cs.stanford.edu/downloads/rt_benchmark"
    for task in lift can square; do
        dest="$ROBO_BASE/$task/ph/low_dim.hdf5"
        download_file "$BASE_URL/$task/ph/low_dim_v141.hdf5" "$dest"
    done
fi

# 轉換成 abs（訓練需要）
for task in lift can square transport; do
    raw="$ROBO_BASE/$task/ph/low_dim.hdf5"
    [[ -f "$raw" ]] && convert_robomimic "$task/ph" "$raw"
done

# =============================================================================
# 2. PUSHT — 2D 推塊基準
# =============================================================================
header "PushT (2D baseline)"

PUSHT_DIR="$DATA_ROOT/pusht"
mkdir -p "$PUSHT_DIR"

PUSHT_ZIP="$PUSHT_DIR/pusht.zarr.zip"
download_file \
    "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zarr.zip" \
    "$PUSHT_ZIP"

if [[ ! -d "$PUSHT_DIR/pusht.zarr" ]]; then
    info "解壓縮 pusht.zarr.zip …"
    unzip -q "$PUSHT_ZIP" -d "$PUSHT_DIR"
    ok "$PUSHT_DIR/pusht.zarr"
else
    skip "$PUSHT_DIR/pusht.zarr"
fi

# =============================================================================
# 3. MIMICGEN — 夾爪任務合成擴增示範
# =============================================================================
header "MimicGen (合成示範擴增 + 夾爪)"

MIMICGEN_BASE="$DATA_ROOT/mimicgen"

if python -c "import robomimic" &>/dev/null 2>&1; then
    info "下載 MimicGen 夾爪任務 (lift_d0 / can_d0 / square_d0) …"
    # MimicGen 1000-demo 版本；--tasks 接受 mimicgen_environments 的任務名
    # 官方使用 robomimic downloader 的 mimicgen 路徑
    python -m robomimic.scripts.download_datasets \
        --tasks lift_d0 can_d0 square_d0 \
        --dataset_types mg \
        --hdf5_types low_dim \
        --download_dir "$MIMICGEN_BASE" || \
    warn "MimicGen 部分任務不在官方 robomimic downloader，請手動從 HuggingFace 下載：
       huggingface-cli download amandlek/mimicgen_data --local-dir $MIMICGEN_BASE"
else
    warn "robomimic 未安裝，手動下載 MimicGen：
       pip install huggingface_hub
       huggingface-cli download amandlek/mimicgen_data --local-dir $MIMICGEN_BASE"
fi

# =============================================================================
# 4. MANISKILL2 — 夾爪精準操作
# =============================================================================
header "ManiSkill2 (MuJoCo + 夾爪)"

if python -c "import mani_skill2" &>/dev/null 2>&1; then
    info "下載 ManiSkill2 夾爪任務資產 …"
    # PickCube、StackCube、PegInsertionSide 均為平行夾爪
    for uid in PickCube-v0 StackCube-v0 PegInsertionSide-v0 TurnFaucet-v0; do
        python -m mani_skill2.utils.download_asset --uid "$uid" \
            --output-dir "$DATA_ROOT/maniskill2" || \
        warn "  $uid 下載失敗，請稍後重試"
    done
    # 示範資料
    python -m mani_skill2.utils.download_demo --uid PickCube-v0 \
        --output-dir "$DATA_ROOT/maniskill2/demos" || true
    ok "ManiSkill2 完成"
else
    warn "mani_skill2 未安裝：pip install mani-skill2==0.5.3
  安裝後重新執行此腳本即可下載"
fi

# =============================================================================
# 5. META-WORLD — 不需要下載資料集
# =============================================================================
header "Meta-World (50-task benchmark)"
info "Meta-World 為模擬 benchmark library，不需要預下載資料集"
info "安裝：pip install metaworld @ git+https://github.com/Farama-Foundation/Metaworld"
info "資料在訓練時由 env.step() 即時生成（或用 SAC 收集示範）"

# =============================================================================
# 6. BRIDGE V2 — 廚房物件抓放（夾爪末端，~10 TB，選填）
# =============================================================================
header "Bridge V2 (可選，需 --with-bridge)"

BRIDGE_DIR="$DATA_ROOT/bridge_v2"

if [[ "$WITH_BRIDGE" == true ]]; then
    mkdir -p "$BRIDGE_DIR"
    if command -v gsutil &>/dev/null; then
        info "使用 gsutil 下載 Bridge V2 (~10 TB) …"
        gsutil -m rsync -r \
            gs://rail-datasets/bridge_release/data/ "$BRIDGE_DIR/"
        ok "Bridge V2 → $BRIDGE_DIR"
    else
        warn "未找到 gsutil，改用 HuggingFace Hub …"
        if python -c "import huggingface_hub" &>/dev/null 2>&1; then
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='rail-berkeley/bridge_dataset',
    repo_type='dataset',
    local_dir='$BRIDGE_DIR',
    ignore_patterns=['*.tfrecord.zst'],   # 先跳過超大壓縮檔
)
print('Bridge V2 metadata 下載完成')
"
        else
            warn "請安裝 huggingface_hub：pip install huggingface_hub
  然後執行：
    python -c \"from huggingface_hub import snapshot_download; \\
        snapshot_download(repo_id='rail-berkeley/bridge_dataset', \\
        repo_type='dataset', local_dir='$BRIDGE_DIR')\""
        fi
    fi
else
    info "Bridge V2 已跳過（加上 --with-bridge 旗標下載，約 10 TB）"
fi

# =============================================================================
# 7. DROID — 多樣夾爪抓取（~15 TB，選填）
# =============================================================================
header "DROID (可選，需 --with-droid)"

DROID_DIR="$DATA_ROOT/droid"

if [[ "$WITH_DROID" == true ]]; then
    mkdir -p "$DROID_DIR"
    info "使用 HuggingFace Hub 下載 DROID (~15 TB) …"
    if python -c "import huggingface_hub" &>/dev/null 2>&1; then
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='physical-intelligence/droid',
    repo_type='dataset',
    local_dir='$DROID_DIR',
)
print('DROID 下載完成')
"
        ok "DROID → $DROID_DIR"
    else
        warn "請安裝：pip install huggingface_hub
  hf_token 需要申請 DROID 資料集存取權：
  https://huggingface.co/datasets/physical-intelligence/droid"
    fi
else
    info "DROID 已跳過（加上 --with-droid 旗標下載，約 15 TB）"
fi

# =============================================================================
# 8. OPEN X-EMBODIMENT (RT-X) — 篩選夾爪（極大，選填）
# =============================================================================
header "Open X-Embodiment / RT-X (可選，需 --with-openx)"

OXE_DIR="$DATA_ROOT/open_x_embodiment"

if [[ "$WITH_OPENX" == true ]]; then
    mkdir -p "$OXE_DIR"
    info "下載 Open X 夾爪子集（bridge_data_v2 + fractal20220817 + kuka）…"
    # 只下載 parallel gripper 相關子資料集，避免下載全部 (>500 TB)
    if python -c "import tensorflow_datasets" &>/dev/null 2>&1; then
        python - <<'PY'
import tensorflow_datasets as tfds
import os

OXE_DIR = os.environ.get("OXE_DIR", "data/open_x_embodiment")

# 夾爪相容的子資料集（較小）
GRIPPER_DATASETS = [
    "bridge",                    # Bridge / BridgeV2   ~1.8 TB
    "fractal20220817_data",      # RT-1               ~1.5 TB
    "kuka",                      # KUKA grasping      ~0.7 TB
    "taco_play",                 # TACO-Play          ~0.4 TB
    "jaco_play",                 # Jaco gripper       ~0.2 TB
    "berkeley_cable_routing",    # Cable routing       ~0.1 TB
]

for name in GRIPPER_DATASETS:
    print(f"[下載] {name} …")
    try:
        ds, info = tfds.load(name, data_dir=OXE_DIR, with_info=True)
        print(f"  OK: {info.splits}")
    except Exception as e:
        print(f"  [warn] {name}: {e}")
PY
    else
        warn "請安裝：pip install tensorflow-datasets
  建議只下載特定子集，完整 Open X 超過 500 TB"
    fi
else
    info "Open X 已跳過（加上 --with-openx 旗標下載夾爪子集，約 4-5 TB）"
fi

# =============================================================================
# 9. ADROIT — 跳過（需靈巧手，非夾爪）
# =============================================================================
header "Adroit (略過 — 需 24-DOF 靈巧手)"
warn "Adroit 任務（pen / hammer / relocate）需要 24 自由度靈巧手"
warn "夾爪使用者請跳過，或僅使用 door-v1（door 任務可用夾爪）"
info "若需 door 任務：pip install d4rl
  python -c \"import gym, d4rl; gym.make('door-human-v1')\""

# =============================================================================
# 10. 摘要
# =============================================================================
header "下載完成摘要"

echo ""
echo "  資料根目錄：$DATA_ROOT"
echo ""
printf "  %-30s %s\n" "資料集" "路徑"
printf "  %-30s %s\n" "------" "----"

_check() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        printf "  \033[1;32m✓\033[0m %-28s %s\n" "$label" "$path"
    else
        printf "  \033[0;90m✗\033[0m %-28s %s\n" "$label" "$path (未下載)"
    fi
}

_check "Robomimic lift/ph abs"   "$ROBO_BASE/lift/ph/low_dim_abs.hdf5"
_check "Robomimic can/ph abs"    "$ROBO_BASE/can/ph/low_dim_abs.hdf5"
_check "Robomimic square/ph abs" "$ROBO_BASE/square/ph/low_dim_abs.hdf5"
_check "PushT zarr"              "$PUSHT_DIR/pusht.zarr"
_check "MimicGen lift_d0"        "$MIMICGEN_BASE/lift_d0/ph/low_dim.hdf5"
_check "ManiSkill2 assets"       "$DATA_ROOT/maniskill2"
_check "Bridge V2"               "$BRIDGE_DIR"
_check "DROID"                   "$DROID_DIR"
_check "Open X-Embodiment"       "$OXE_DIR"

echo ""
info "訓練指令："
echo "  python train.py --config-name=lift_ph    # Robomimic Lift（推薦起點）"
echo "  python train.py --config-name=pusht      # PushT 2D baseline"
echo ""
