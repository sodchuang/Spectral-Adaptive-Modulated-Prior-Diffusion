ㄏ2A 熱啟動** 與 **Modulated Prior** 整合進頻域流程。DCT 作為頻率切割工具（後續版本預計升級為 FFT 分離高低頻）。

---

### 訓練流程圖 (Training Pipeline)

```mermaid
flowchart LR
    subgraph TIME1["時域 Time Domain"]
        direction TB
        OBS_IN["觀測序列\nRobot State (19-dim)"]
        GT["GT 動作序列 A\n(B, H=16, Da=10)"]
        AT1["前一幀動作 A_prev"]
    end

    subgraph PRIOR_BLOCK["A2A 先驗建構 (warm-start)"]
        direction TB
        AT1 --> DCT_P["DCT(A_prev)"]
        DCT_P --> X0["x_0 = DCT_prev + σ·ε\n知情初始化"]
    end

    subgraph FM_BLOCK["Flow Matching 插值"]
        direction TB
        GT --> DCT_GT["DCT(A) = x_1"]
        X0 --> FM_INT["x_t = (1-t)·x_0 + t·x_1"]
        DCT_GT --> FM_INT
        FM_INT --> TARGET["target velocity\nu = x_1 − x_0"]
    end

    subgraph MODEL_BLOCK["SampNet (MAE Transformer)"]
        direction TB
        FP_ENC["Encoder\nx_t + t_embed + cond"]
        FP_DEC["Decoder + flow_head"]
        FP_ENC --> FP_DEC
        FP_DEC --> VPRED["v_pred"]
    end

    subgraph LOSS_BLOCK["Loss"]
        LOSS_OUT["L = ‖v_pred − u‖²"]
    end

    OBS_IN --> MODEL_BLOCK
    FM_INT --> FP_ENC
    VPRED --> LOSS_OUT
    TARGET --> LOSS_OUT

    style TIME1 fill:#e8f4fd,stroke:#4a90d9
    style PRIOR_BLOCK fill:#fff3e0,stroke:#f5a623
    style FM_BLOCK fill:#e8f8e8,stroke:#4caf50
    style MODEL_BLOCK fill:#f3e5f5,stroke:#9c27b0
    style LOSS_BLOCK fill:#fce4ec,stroke:#e91e63
```

---

### 推論流程圖 (Inference Pipeline)

```mermaid
flowchart LR
    subgraph ENV["MuJoCo 環境"]
        direction TB
        OT["當前觀測 oₜ\nRobot State (19-dim)"]
        AT_PREV["上一幀動作 A_prev\n(warm-start buffer)"]
        ROBOT["執行動作\n8 steps per call"]
    end

    subgraph WARMSTART["A2A 初始化"]
        AT_PREV --> DCT2["DCT(A_prev) + σ·ε"]
        DCT2 --> X0I["x_0 (informed)"]
    end

    subgraph ODE["Euler ODE (6 steps)"]
        direction TB
        X0I --> STEP["x_{t+dt} = x_t + dt·v_θ(x_t,t,c)"]
        STEP --> STEP
        STEP --> X1["x_1 (最終頻域結果)"]
    end

    subgraph DECODE["解碼"]
        X1 --> IDCT["iDCT → actions\n(B, H=16, Da=10)"]
        IDCT --> SLICE["切片\n[:, To-1 : To-1+8]"]
    end

    OT --> ODE
    SLICE --> ROBOT
    SLICE --> AT_PREV

    style ENV fill:#e8f4fd,stroke:#4a90d9
    style WARMSTART fill:#fff3e0,stroke:#f5a623
    style ODE fill:#e8f8e8,stroke:#4caf50
    style DECODE fill:#f3e5f5,stroke:#9c27b0

    subgraph PRIOR3["FFT 熱啟動先驗\n(A2A + MPD)"]
        direction TB
        AT_PREV --> FFT3["FFT(A_prev)"]
        FFT3 --> LOW3["低頻 F_low\n+ σ·ε"]
        FFT3 --> HIGH3["高頻\n→ ε ~ N(0,I)"]
        LOW3 --> IFFT3["IFFT\n初始先驗 x_T"]
        HIGH3 --> IFFT3
    end

    subgraph ITER_BLOCK["迭代生成 Iter 1 → N_iter\n(FreqPolicy Sampling)"]
        direction TB
        ITER_IN["輸入: x_T + masked tokens\nindex = l₀ (低頻起點)"]
        ENC_I2["FreqPolicy Encoder"]
        DEC_I2["FreqPolicy Decoder"]
        ODE["Flow Matching ODE\n1~3 步去噪"]
        IFFT_I2["IFFT → k-level token\n更新 index → lᵢ₊₁"]
        ITER_IN --> ENC_I2 --> DEC_I2 --> ODE --> IFFT_I2
        IFFT_I2 -->|"重複 N_iter 次\n頻率逐步提升"| ENC_I2
    end

    subgraph OUT_BLOCK["輸出還原"]
        direction TB
        FULL2["全頻 token (Full-frequency)"]
        IFFT_OUT2["IFFT → 時域動作 Aₜ\n(n × 9)"]
        FULL2 --> IFFT_OUT2
    end

    C2 --> ITER_IN
    IFFT3 --> ITER_IN
    IFFT_I2 --> FULL2
    IFFT_OUT2 --> ROBOT
    IFFT_OUT2 -->|"存為 A_prev\nA2A 閉環"| AT_PREV

    style TIME2 fill:#e8f4fd,stroke:#4a90d9
    style PRIOR3 fill:#fff3e0,stroke:#f5a623
    style ITER_BLOCK fill:#f3e5f5,stroke:#9c27b0
    style OUT_BLOCK fill:#e8f8e8,stroke:#4caf50
```

---

### 三大技術整合說明

| 技術來源 | 整合位置 | 作用 |
| :--- | :--- | :--- |
| **Modulated Prior Diffusion** | 先驗建構（PRIOR 區塊）| 以 $A_{t-1}$ 低頻係數作為先驗均值，取代純高斯起點 |
| **A2A Flow Matching** | 熱啟動先驗 + ODE 去噪 | 低頻熱啟動 + 1~3 步 ODE，取代 50 步標準去噪 |
| **FreqPolicy（主幹）** | Encoder–Decoder + 迭代生成 | FFT 切割頻率層、Transformer 編解碼、per-token diffusion loss |

---


## 設計哲學：為什麼要這樣做？ (Design Philosophy)

### 1. 解決「延遲」
利用 **A2A** 知情初始化配合 **Modulated Prior**，將推論步數壓縮至 **1-3 步**，解決標準 Diffusion 運算過慢之痛點。

### 2. 解決「抖動」
在頻域生成動作等於是在底層進行物理級的「低通濾波」，從數學本質上確保產出軌跡的連貫性與絲滑度。

### 3. 解決「反應遲鈍」
透過**頻譜差異化**策略，使低頻（大方向）靠記憶維持穩定，高頻（細微修正）由當前視覺感官引導去噪，讓機器人兼具肌肉記憶與靈敏反應。
