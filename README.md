# Spectral-Adaptive-Modulated-Prior-Diffusion
# SAMP-Diff: Spectral-Adaptive Modulated Prior Diffusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本專案實作了一套結合 **A2A (Action-to-Action)** 效率、**FreqPolicy** 物理連貫性，以及 **Modulated Prior** 數學框架的機器人控制策略。透過將動作空間轉移至頻域 (Frequency Domain)，我們實現了具備「肌肉記憶」且能即時反應的工業級控制。

## 核心創新：頻譜差異化調變 (Spectral-Selective Modulation)

本研究解決了傳統 A2A 在時域處理高頻噪音時導致的「過度平滑」問題。我們提出針對不同頻譜成分實施差異化策略：

* **低頻成分 (Low-Freq)**：動作的骨架與慣性。套用 **A2A + Modulated Prior**，實現 1-Step 高速推論。
* **高頻成分 (High-Freq)**：動作的修正與細節。維持 **Standard Diffusion Denoising**，保留機器人對動態環境的靈敏反應能力。

$$x_T = [\mu_{low} + \sigma_{low} \odot \epsilon_{low}, \epsilon_{high}]$$

## 支援環境與資料集 (Supported Benchmarks)

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

## 預測架構 (Prediction Architecture)

SAMP-Diff 的預測流程是一個從「感官」到「頻譜決策」再回到「物理執行」的完整閉環，其核心邏輯在於將動作生成從不穩定的時域移轉至具備物理意義的頻域空間：

1.  **多模態編碼 (Multimodal Encoding)**：
    系統同步接收當前視覺影像（Vision）與語言指令（Language），透過預訓練之編碼器提取環境特徵（Observation Embedding）。

2.  **時頻轉換 (Spectral Mapping)**：
    讀取歷史動作序列 $A_{t-1}$，利用 **DCT (離散餘弦變換)** 將其從「時域」轉移至「頻域」空間，取得頻率係數 $C_{t-1}$。

3.  **頻譜差異化調變 (Asymmetric Spectral Modulation)**：
    * **低頻區塊 (Low-Freq)**：將 $C_{t-1, \text{low}}$ 丟入 **Prior Network (秋賢學長之調變技術)**，預測出當前動作的先驗分佈參數 $\mu, \sigma$。
    * **高頻區塊 (High-Freq)**：不參考歷史動作，直接賦予標準高斯隨機雜訊，確保去噪過程具備探索空間。

4.  **快速頻域去噪 (Spectral Denoising)**：
    擴散模型以調變後的先驗分布為起點，在頻率空間執行 1-3 步的快速採樣（Flow Matching / Diffusion），生成目標頻率係數 $C_t$。

5.  **動作還原與執行 (Inverse Mapping & Actuation)**：
    透過 **IDCT (逆離散餘弦變換)** 將生成的係數還原為時間軸動作 $A_t$，經由 RTDE 通訊協定傳送至 UR 機器人執行。

---

## 設計哲學：為什麼要這樣做？ (Design Philosophy)

本架構並非盲目組合技術，而是針對工業機器人在「效率」、「絲滑」與「靈敏」之間的物理矛盾提出的最佳解法。

### 1. 為什麼要用 A2A + 秋賢學長的技術？ (解決「延遲」)
* **問題**：標準擴散模型從純雜訊開始採樣，通常需要 50 步以上才能收斂，這會造成機器人大腦嚴重的推論延遲。
* **對策**：利用 **A2A** 的知情初始化 (Informed Initialization) 概念，配合 **Modulated Prior** 數學框架，直接將起始雜訊推向正確答案的統計範圍。
* **結果**：將推論步數壓縮至 **1-3 步**，實現大於 50Hz 的實時控制。

### 2. 為什麼要在 Frequency Domain 運作？ (解決「抖動」)
* **問題**：時域預測容易在動作區間（Action Chunk）的銜接處產生數值跳變，導致 UR 機器人產生高頻震動。
* **對策**：頻域係數具備能量集中特性，低頻係數即代表了動作的流暢大輪廓。在頻域生成動作，等於是在數學底層進行了物理級的「低通濾波」。
* **結果**：產出的軌跡天生具備工業級的**連貫性與絲滑度**。

### 3. 為什麼高頻要「拒絕」A2A？ (解決「反應遲鈍」)
* **問題**：如果全頻段都強行參考過去（傳統 A2A），當環境發生突發變化時，機器人會因為過度依賴歷史動作而反應遲鈍，產生所謂的「動作滯後」。
* **對策**：提出**頻譜差異化**。低頻（大方向）靠記憶維持穩定，高頻（細微修正）則回歸擴散模型的隨機性，由當前視覺感官引導去噪。
* **結果**：機器人具備了**「有肌肉記憶（低頻），但反射神經依然敏銳（高頻）」**的特性，能處理動態避障與即時修正。


