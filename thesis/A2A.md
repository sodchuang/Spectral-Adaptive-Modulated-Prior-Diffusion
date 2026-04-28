# Action-to-Action (A2A)

## 論文概覽
* **全名**：Action-to-Action Flow Matching: Informed Initialization for High-Frequency Robot Control
* **核心定位**：極速推論與時間連續性優化

## 核心概念：知情初始化 (Informed Initialization)
A2A 的設計哲學在於利用動作序列的時間相關性。在連續的動作任務中，上一個時刻的動作序列 $A_{t-1}$ 與當前動作 $A_t$ 具有極強的統計相關性，系統不應在每一幀都從無意義的雜訊開始推論。

* **傳統方式**：從純隨機高斯雜訊 $x_T \sim \mathcal{N}(0, I)$ 開始去噪。
* **A2A 方式**：將 $A_{t-1}$ 作為起始點，僅加入微小擾動進行「熱啟動」。

$$x_T = A_{t-1} + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

## 本專案之參考價值
A2A 證明了利用歷史資訊能大幅壓縮採樣步數 (NFE)。SAMP-Diff 吸取其效率優勢，但將應用空間從時域遷移至頻域，旨在修正 A2A 在動態環境中反應遲鈍（Over-smoothing）的物理缺陷。

## 局限性與挑戰
* **動作滯後**：由於在全頻段參考過去，導致機器人遇到突發干擾時，會因過度依賴歷史動作而無法即時轉向。
* **時域不連續**：在動作區間銜接處仍可能出現數值跳變產生的微小抖動。
