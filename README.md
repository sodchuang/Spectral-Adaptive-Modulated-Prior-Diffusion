# Spectral-Adaptive-Modulated-Prior-Diffusion
# SAMP-Diff: Spectral-Adaptive Modulated Prior Diffusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本專案實作了一套結合 **A2A (Action-to-Action)** 效率、**FreqPolicy** 物理連貫性，以及 **Modulated Prior** 數學框架的機器人控制策略。透過將動作空間轉移至頻域 (Frequency Domain)，我們實現了具備「肌肉記憶」且能即時反應的工業級控制。

## 🌟 核心創新：頻譜差異化調變 (Spectral-Selective Modulation)

本研究解決了傳統 A2A 在時域處理高頻噪音時導致的「過度平滑」問題。我們提出針對不同頻譜成分實施差異化策略：

* **低頻成分 (Low-Freq)**：動作的骨架與慣性。套用 **A2A + Modulated Prior**，實現 1-Step 高速推論。
* **高頻成分 (High-Freq)**：動作的修正與細節。維持 **Standard Diffusion Denoising**，保留機器人對動態環境的靈敏反應能力。

$$x_T = [\mu_{low} + \sigma_{low} \odot \epsilon_{low}, \epsilon_{high}]$$

## 📂 支援環境與資料集 (Supported Benchmarks)

本專案高度集成，支援以下 10 個主流機器人學習資料集：

| 分類 | 名稱 | 測試重點 |
| :--- | :--- | :--- |
| **基礎驗證** | `PushT` | 2D 軌跡快速迭代 |
| **模仿學習** | `Robomimic` | 標準動作生成基準 |
| **工業大數據** | `Bridge V2` | 真實廚房多任務驗證 |
| **高頻控制** | `DROID` | 視覺引導與靈敏度測試 |
| **精密交互** | `ManiSkill2` | 幾何精度與物理接觸細節 |
| **多任務通用** | `Meta-World` | 跨任務（50種）調變穩定性 |
| **靈巧手控制** | `Adroit` | 24 自由度高維度協調挑戰 |
| **通用大模型** | `Open X (RT-X)` | 跨機器人基礎模型泛化 |
| **數據增強** | `MimicGen` | 合成示範數據擴增實驗 |
| **實體落地** | `UR_Real_Data` | 實驗室 UR 機器人 RTDE 部署 |

## 🏗️ 系統架構



## 🛠️ 安裝說明

```bash
# 建立專屬環境
conda create -n samp_diff python=3.9
conda activate samp_diff

# 安裝核心依賴
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt
