# Spectral-Adaptive Modulated Prior Diffusion (SAMP-Diff)

結合 **A2A 知情初始化**、**頻域（DCT）動作生成** 與 **Flow Matching** 的機器人控制策略。
將動作預測移至頻域，以 6 步 Euler ODE 取代標準 50 步去噪，實現 50Hz+ 即時控制。

---

## 文件索引

| 文件 | 說明 |
| :--- | :--- |
| [README.md](./README.md) | 安裝、訓練、評估（本頁） |
| [PLAN.md](./PLAN.md) | 研究計畫、核心創新、實驗設計、架構圖、設計哲學 |
| [thesis/A2A.md](./thesis/A2A.md) | A2A Flow Matching 深度技術分析 |
| [thesis/DP4.md](./thesis/DP4.md) | Diffusion Policy 4 深度技術分析 |

---

## 環境安裝

```bash
cd SAMP_Diff_v1
conda env create -f conda_environment.yaml
conda activate robodiff
pip install -e .
pip install torchcfm torch-dct
```

> 需要 MuJoCo 授權並完成 [robomimic 資料集下載](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)。

---

## 資料集準備

### 一鍵下載（Linux，推薦）

```bash
cd SAMP_Diff_v1
chmod +x scripts/download_all_datasets.sh

# 標準資料集（Robomimic + PushT + MimicGen + ManiSkill2，約 5-10 GB）
./scripts/download_all_datasets.sh

# 加上大型資料集（選填）
./scripts/download_all_datasets.sh --with-bridge   # Bridge V2  ~10 TB
./scripts/download_all_datasets.sh --with-droid    # DROID      ~15 TB
./scripts/download_all_datasets.sh --with-openx    # Open X 夾爪子集 ~5 TB
./scripts/download_all_datasets.sh --all-large     # 以上全部
```

### Python 下載腳本（跨平台）

```bash
cd SAMP_Diff_v1

python download_data.py                    # lift（預設）
python download_data.py --task can         # 撿鋁罐放入桶
python download_data.py --task square      # 螺帽套入螺柱
python download_data.py --skip-download    # 已下載，只做格式轉換
```

腳本自動完成：下載 `low_dim.hdf5` → 轉換為絕對座標 `low_dim_abs.hdf5`

### 支援資料集

| 分類 | 名稱 | 夾爪相容 |
| :--- | :--- | :---: |
| 基礎驗證 | `PushT` | ✓ |
| 模仿學習 | `Robomimic` lift / can / square / transport | ✓ |
| 數據增強 | `MimicGen` lift_d0 / can_d0 / square_d0 | ✓ |
| 工業大數據 | `Bridge V2` | ✓ |
| 高頻控制 | `DROID` | ✓ |
| 幾何精度 | `ManiSkill2` PickCube / StackCube / PegInsertion | ✓ |
| 多任務通用 | `Meta-World`（不需下載，訓練時即時生成） | ✓ |
| 通用大模型 | `Open X (RT-X)` 夾爪子集 | ✓ |
| 實體落地 | `UR_Real_Data`（自行錄製）| ✓ |
| 靈巧手 | `Adroit` | ✗ |

---

## 訓練

```bash
cd SAMP_Diff_v1

# Robomimic 夾爪任務
python train.py --config-name=lift_ph       # 夾取方塊（推薦入門）
python train.py --config-name=can_ph        # 撿鋁罐
python train.py --config-name=square_ph     # 螺帽套柱
python train.py --config-name=transport_ph  # 雙臂搬運

# MimicGen 合成示範（資料量更多）
python train.py --config-name=mimicgen_lift_d0
python train.py --config-name=mimicgen_can_d0
python train.py --config-name=mimicgen_square_d0

# 多資料集合併訓練
python train.py --config-name=multi_gripper

# 2D 基準
python train.py --config-name=pusht

# 覆蓋參數
python train.py --config-name=lift_ph \
    training.device=cuda:1 \
    dataloader.batch_size=128 \
    training.num_epochs=1000
```

訓練輸出：

```
data/outputs/samp_lowdim_lift_ph/
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=xxxx-test_mean_score=x.xxx.ckpt
├── logs.json.txt
└── wandb/
```

---

## 評估

```bash
cd SAMP_Diff_v1

python eval.py \
    --checkpoint data/outputs/samp_lowdim_lift_ph/checkpoints/latest.ckpt \
    --output_dir data/eval_output/lift_ph \
    --device cuda:0
```

評估輸出：

```
data/eval_output/lift_ph/
├── eval_log.json      ← test/mean_score, train/mean_score
└── media/             ← 錄影片段（.mp4）
```

部署呼叫週期：

```python
policy.reset()                            # 切換 episode 前清除 warm-start buffer
while not done:
    obs_dict = env.get_obs()
    result   = policy.predict_action(obs_dict)
    action   = result['action']           # shape: (1, 8, 10)
    env.step(action[0])
```

---

## 研究計畫與架構細節

核心創新、訓練/推論流程圖、實驗設計、設計哲學、v1→v2 路線圖請見：

**[→ PLAN.md](./PLAN.md)**
