# HW2｜多元線性回歸 × CRISP-DM

以 Kaggle 實際資料集進行多元線性回歸（Multiple Linear Regression）分析，完整走過 CRISP-DM 六個階段，包含特徵選擇、模型評估，以及帶 **信賴區間／預測區間** 的預測圖，並以 Streamlit 部署成互動網頁。

- 🌐 線上展示：`<Streamlit 網址>`
- 📄 報告：[report.md](report.md)（含「GPT 輔助內容」與「NotebookLM 摘要」）

## 資料集

| 問題 | 來源 | 樣本 | 特徵 | 目標 |
|---|---|---|---|---|
| WHO 平均壽命 | [Kaggle: Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) | 2,938 | 21 | `Life expectancy` |
| 洪水機率 | [Kaggle: Flood Prediction Dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset) | 3,000 | 20 | `FloodProbability` |

## 功能

- **下拉選單切換兩種問題**（加分項 1）
- **可輸入任意 CSV 網址**作為資料集，例如 [50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)；可自選目標欄位與排除欄位，類別欄位會自動 One-Hot 編碼（加分項 2）
- 四種特徵選擇：Forward Stepwise、RFE、Lasso、SelectKBest，並以 5-fold CV 曲線建議 k
- OLS 係數表（含 p 值與信賴區間）、MAE / MSE / RMSE / R² / Adjusted R²
- 預測圖（CI 與 PI 帶狀圖）、預測區間覆蓋率、殘差圖、Q-Q 圖、Durbin–Watson / Jarque–Bera / Breusch–Pagan 檢定、VIF
- 與全部特徵的 OLS、Ridge、Lasso 比較
- 部署分頁：輸入新樣本即時預測，並顯示單一特徵效果曲線

## 主要結果（WHO 平均壽命，依國家分組切分）

| 模型 | 特徵數 | 測試 RMSE | 測試 R² |
|---|---|---|---|
| OLS 全部特徵 | 20 | 6.73 | 0.515 |
| **OLS + Forward Stepwise** | **11** | **4.09** | **0.821** |
| Lasso（CV） | 20 | 4.06 | 0.824 |

95% 預測區間的測試覆蓋率為 93.4%。詳見 [report.md](report.md)。

![prediction](figures/life_prediction_interval.png)

## 執行方式

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # 互動網頁
python analysis.py            # 重新產生 figures/ 與 results.json
pip install pytest && pytest  # 單元測試
```

## 專案結構

```text
HW2/
├── app.py          # Streamlit 介面（CRISP-DM 六個分頁）
├── mlr.py          # 前處理、特徵選擇、OLS 區間估計、評估
├── analysis.py     # 產生報告圖表與數據
├── report.md       # 作業報告
├── data/           # Kaggle 資料集（CSV）
├── figures/        # 報告圖表與 results.json
├── chat_logs/      # AI 對話紀錄 PDF
└── tests/          # pytest
```
