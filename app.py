"""HW2: Multiple linear regression following CRISP-DM (Streamlit app)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import mlr

DEFAULT_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
CUSTOM = "custom"

st.set_page_config(page_title="HW2 多元線性回歸｜CRISP-DM", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="讀取資料中…")
def load(source: str) -> pd.DataFrame:
    return mlr.load_dataset(source)


@st.cache_data(show_spinner="準備資料中…")
def prepare(df, target, drop, group, test_size, seed):
    return mlr.prepare_data(df, target, list(drop), group, test_size, seed)


@st.cache_data(show_spinner="特徵排序與交叉驗證中…")
def rank_and_curve(X, y, method, seed):
    ranking = mlr.rank_features(X, y, method, seed)
    return ranking, mlr.k_curve(X, y, ranking, seed)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ 設定")
problem_key = st.sidebar.selectbox(
    "選擇問題（Problem）",
    [*mlr.PROBLEMS, CUSTOM],
    format_func=lambda k: mlr.PROBLEMS[k]["label"] if k in mlr.PROBLEMS else "自訂資料集（輸入 CSV 網址）",
)

if problem_key == CUSTOM:
    url = st.sidebar.text_input("CSV 網址（raw）", DEFAULT_URL)
    try:
        df = load(url)
    except Exception as exc:  # noqa: BLE001 - show any read error to the user
        st.error(f"無法讀取資料集：{exc}")
        st.stop()
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        st.error("資料集中沒有數值欄位可作為回歸目標。")
        st.stop()
    target = st.sidebar.selectbox("目標欄位（y）", numeric, index=len(numeric) - 1)
    drop = st.sidebar.multiselect("排除欄位（ID、名稱等）", [c for c in df.columns if c != target])
    problem = {
        "label": "自訂資料集",
        "kaggle": url,
        "target": target,
        "drop": drop,
        "group": None,
        "unit": "",
        "business": f"使用者提供的資料集，以其餘欄位預測「{target}」。",
        "success": "測試集 R² 越高越好，並檢查預測區間的覆蓋率。",
    }
else:
    problem = mlr.PROBLEMS[problem_key]
    df = load(str(problem["path"]))
    target = problem["target"]

test_size = st.sidebar.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
seed = int(st.sidebar.number_input("隨機種子", value=42, step=1))
method_label = st.sidebar.selectbox("特徵選擇方法", list(mlr.SELECTION_METHODS))
method = mlr.SELECTION_METHODS[method_label]
conf = st.sidebar.select_slider("信賴水準", [0.80, 0.90, 0.95, 0.99], value=0.95, format_func=lambda v: f"{v:.0%}")
alpha = 1 - conf

try:
    prep = prepare(df, target, tuple(problem["drop"]), problem["group"], test_size, seed)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if len(prep.X_train) <= prep.X_train.shape[1] + 1:
    st.error("訓練樣本數不足以估計所有係數，請換一個資料集或降低測試集比例。")
    st.stop()

ranking, curve = rank_and_curve(prep.X_train, prep.y_train, method, seed)
suggested = mlr.suggest_k(curve)
k = st.sidebar.slider("選取特徵數 k", 1, len(ranking), suggested, help=f"建議值 {suggested}：CV RMSE 與最佳值相差 1% 以內的最小 k")
selected = ranking[:k]

result = mlr.fit_ols(prep.X_train[selected], prep.y_train)
test_iv = mlr.intervals(result, prep.X_test[selected], alpha)
test_metrics = mlr.metrics(prep.y_test, test_iv["pred"], k)
train_metrics = mlr.metrics(prep.y_train, result.fittedvalues, k)
coverage = float(((prep.y_test >= test_iv["pi_low"]) & (prep.y_test <= test_iv["pi_high"])).mean())

st.sidebar.markdown("---")
st.sidebar.caption("資料來源")
st.sidebar.markdown(f"[{problem['kaggle']}]({problem['kaggle']})")

# ---------------------------------------------------------------- header
st.title("📈 多元線性回歸 × CRISP-DM")
st.caption(f"問題：{problem['label']}　｜　目標 y：`{target}`　｜　特徵選擇：{method_label}，k = {k}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("測試集 R²", f"{test_metrics['R2']:.3f}")
c2.metric("測試集 RMSE", f"{test_metrics['RMSE']:.4g} {problem['unit']}")
c3.metric("Adjusted R²", f"{test_metrics['Adj_R2']:.3f}")
c4.metric(f"{conf:.0%} 預測區間覆蓋率", f"{coverage:.1%}")

tabs = st.tabs(
    ["1️⃣ Business Understanding", "2️⃣ Data Understanding", "3️⃣ Data Preparation", "4️⃣ Modeling", "5️⃣ Evaluation", "6️⃣ Deployment"]
)

# ---------------------------------------------------------------- 1. business
with tabs[0]:
    st.subheader("商業理解")
    st.write(problem["business"])
    st.markdown(
        f"""
- **預測目標**：`{target}`（連續數值 → 回歸問題）
- **模型**：多元線性回歸 $y = \\beta_0 + \\beta_1x_1 + \\dots + \\beta_kx_k + \\varepsilon$（OLS 估計）
- **成功標準**：{problem['success']}
- **資料來源**：[{problem['kaggle']}]({problem['kaggle']})
"""
    )

# ---------------------------------------------------------------- 2. data understanding
with tabs[1]:
    st.subheader("資料理解")
    n_features = df.shape[1] - 1 - len(problem["drop"])
    a, b, c = st.columns(3)
    a.metric("樣本數", f"{len(df):,}")
    b.metric("原始特徵數（不含 y）", df.shape[1] - 1)
    c.metric("缺失值總數", int(df.isna().sum().sum()))
    st.dataframe(df.head(20), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**目標變數分布**")
        st.plotly_chart(px.histogram(df, x=target, nbins=40), width="stretch")
    with right:
        st.markdown("**各欄位缺失比例**")
        miss = df.isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            st.success("沒有缺失值。")
        else:
            st.plotly_chart(px.bar(x=miss.values, y=miss.index, orientation="h", labels={"x": "缺失比例", "y": ""}), width="stretch")

    st.markdown("**數值欄位相關係數矩陣**")
    corr = df.select_dtypes("number").corr()
    st.plotly_chart(
        px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", height=650),
        width="stretch",
    )
    if target in corr:
        top = corr[target].drop(target).sort_values(key=np.abs, ascending=False).head(10)
        st.markdown("**與目標相關性最高的 10 個欄位**")
        st.dataframe(top.rename("Pearson r").to_frame().style.format("{:.3f}"), width="stretch")
    st.markdown("**描述統計**")
    st.dataframe(df.describe().T, width="stretch")

# ---------------------------------------------------------------- 3. preparation
with tabs[2]:
    st.subheader("資料準備")
    st.markdown("所有統計量（中位數、類別眾數）只用**訓練集**計算，再套用到測試集，避免資料洩漏。")
    for line in prep.log:
        st.markdown(f"- {line}")
    st.markdown(f"- 前處理後可用特徵數：**{prep.X_train.shape[1]}**")
    st.dataframe(prep.X_train.head(10), width="stretch")

# ---------------------------------------------------------------- 4. modeling
with tabs[3]:
    st.subheader("特徵選擇（Feature Selection）")
    st.markdown(
        f"以 **{method_label}** 對訓練集特徵排序，再對前 k 個特徵做 5-fold 交叉驗證，"
        f"選出 CV RMSE 與最佳值相差 1% 以內的最小 k（建議 k = **{suggested}**）。"
    )
    fig = go.Figure()
    fig.add_scatter(x=curve["k"], y=curve["CV_RMSE"], mode="lines+markers", name="CV RMSE", customdata=curve["added"],
                    hovertemplate="k=%{x}<br>RMSE=%{y:.4g}<br>新增：%{customdata}<extra></extra>")
    fig.add_scatter(x=curve["k"], y=curve["CV_Adj_R2"], mode="lines+markers", name="CV Adjusted R²", yaxis="y2")
    fig.add_vline(x=k, line_dash="dash", annotation_text=f"目前 k={k}")
    fig.update_layout(xaxis_title="特徵數 k", yaxis_title="CV RMSE", yaxis2=dict(title="Adjusted R²", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.1), height=420)
    st.plotly_chart(fig, width="stretch")

    rank_df = pd.DataFrame({"排名": range(1, len(ranking) + 1), "特徵": ranking, "是否選取": ["✅" if f in selected else "" for f in ranking]})
    st.dataframe(rank_df, hide_index=True, width="stretch")

    st.subheader("OLS 模型")
    st.latex(r"\hat{y} = " + f"{result.params['const']:.4g}" + "".join(
        f" {'+' if v >= 0 else '-'} {abs(v):.4g}\\,\\text{{{name.replace('_', ' ')}}}" for name, v in result.params.drop("const").items()
    ))
    coef = mlr.coefficient_table(result, alpha)
    st.dataframe(coef.style.format("{:.4g}").map(lambda v: "color: #c0392b" if isinstance(v, float) and v > 0.05 else "", subset=["p_value"]),
                 width="stretch")
    st.caption(f"紅字表示 p > 0.05（該係數在 {conf:.0%} 水準下不顯著）。ci_low / ci_high 為係數的 {conf:.0%} 信賴區間。")
    with st.expander("statsmodels 完整摘要"):
        st.text(result.summary().as_text())

# ---------------------------------------------------------------- 5. evaluation
with tabs[4]:
    st.subheader("模型評估")
    st.dataframe(
        pd.DataFrame({"訓練集": train_metrics, "測試集": test_metrics}).T.style.format("{:.4g}"),
        width="stretch",
    )

    st.markdown(f"### 預測圖（{conf:.0%} 信賴區間 & 預測區間）")
    st.markdown(
        "- **信賴區間（CI）**：平均反應值 $E[y\\mid x]$ 的不確定性，較窄。\n"
        "- **預測區間（PI）**：單一新觀測值 $y$ 的不確定性（再加上誤差變異），較寬。"
    )
    plot_df = test_iv.join(prep.y_test.rename("actual")).sort_values("pred").reset_index(drop=True)
    x = plot_df.index
    fig = go.Figure()
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[plot_df["pi_high"], plot_df["pi_low"][::-1]], fill="toself",
                    fillcolor="rgba(52,152,219,0.18)", line=dict(width=0), name=f"{conf:.0%} 預測區間", hoverinfo="skip")
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[plot_df["ci_high"], plot_df["ci_low"][::-1]], fill="toself",
                    fillcolor="rgba(52,152,219,0.45)", line=dict(width=0), name=f"{conf:.0%} 信賴區間", hoverinfo="skip")
    fig.add_scatter(x=x, y=plot_df["pred"], mode="lines", line=dict(color="#1f4e79", width=2), name="預測值")
    inside = (plot_df["actual"] >= plot_df["pi_low"]) & (plot_df["actual"] <= plot_df["pi_high"])
    fig.add_scatter(x=x[inside], y=plot_df["actual"][inside], mode="markers", marker=dict(size=5, color="#e67e22"), name="實際值（區間內）")
    fig.add_scatter(x=x[~inside], y=plot_df["actual"][~inside], mode="markers", marker=dict(size=7, color="#c0392b", symbol="x"), name="實際值（區間外）")
    fig.update_layout(xaxis_title="測試樣本（依預測值排序）", yaxis_title=target, height=500, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"測試集有 {coverage:.1%} 的實際值落在 {conf:.0%} 預測區間內（理論值 {conf:.0%}）。")

    left, right = st.columns(2)
    with left:
        st.markdown("**實際值 vs 預測值**")
        fig = px.scatter(plot_df, x="actual", y="pred", opacity=0.6, labels={"actual": "實際值", "pred": "預測值"})
        lo, hi = float(min(plot_df["actual"].min(), plot_df["pred"].min())), float(max(plot_df["actual"].max(), plot_df["pred"].max()))
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="y = x")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**殘差 vs 預測值（訓練集）**")
        fig = px.scatter(x=result.fittedvalues, y=result.resid, opacity=0.5, labels={"x": "預測值", "y": "殘差"})
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**殘差分布**")
        st.plotly_chart(px.histogram(x=result.resid, nbins=50, labels={"x": "殘差"}), width="stretch")
    with right:
        st.markdown("**Q-Q 圖**")
        from scipy import stats

        (osm, osr), (slope, intercept, _) = stats.probplot(result.resid, dist="norm")
        fig = px.scatter(x=osm, y=osr, labels={"x": "理論分位數", "y": "樣本分位數"}, opacity=0.5)
        fig.add_scatter(x=osm, y=slope * osm + intercept, mode="lines", line=dict(color="gray"), name="常態參考線")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**回歸假設檢定**")
    st.dataframe(mlr.assumption_tests(result).style.format({"統計量": "{:.3f}", "p 值": "{:.3g}"}), hide_index=True, width="stretch")
    st.markdown("**多重共線性（VIF，> 10 表示嚴重共線）**")
    st.dataframe(mlr.vif_table(prep.X_train[selected]).style.format({"VIF": "{:.2f}"}), hide_index=True, width="stretch")

    st.markdown("**模型比較（測試集）**")
    st.dataframe(mlr.compare_models(prep, selected, seed).style.format({c: "{:.4g}" for c in ["MAE", "MSE", "RMSE", "R2", "Adj_R2"]}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- 6. deployment
with tabs[5]:
    st.subheader("部署：輸入新樣本進行預測")
    st.markdown("調整下列特徵值（預設為訓練集中位數），即時得到預測值與信賴 / 預測區間。")
    values = {}
    cols = st.columns(3)
    for i, feat in enumerate(selected):
        series = prep.X_train[feat]
        with cols[i % 3]:
            if series.nunique() <= 2:
                values[feat] = float(st.selectbox(feat, sorted(series.unique()), index=int(series.median() == series.max()), key=f"in_{feat}"))
            else:
                values[feat] = st.number_input(feat, value=float(series.median()), key=f"in_{feat}", format="%.4g")
    new = pd.DataFrame([values])[selected]
    out = mlr.intervals(result, new, alpha).iloc[0]
    a, b, c = st.columns(3)
    a.metric("預測值", f"{out['pred']:.4g} {problem['unit']}")
    b.metric(f"{conf:.0%} 信賴區間", f"[{out['ci_low']:.4g}, {out['ci_high']:.4g}]")
    c.metric(f"{conf:.0%} 預測區間", f"[{out['pi_low']:.4g}, {out['pi_high']:.4g}]")

    st.markdown("**單一特徵的效果曲線**（其他特徵固定在上方輸入值）")
    feat = st.selectbox("變動的特徵", selected, key="effect_feat")
    grid = np.linspace(prep.X_train[feat].quantile(0.01), prep.X_train[feat].quantile(0.99), 60)
    sweep = pd.concat([new] * len(grid), ignore_index=True)
    sweep[feat] = grid
    sw = mlr.intervals(result, sweep, alpha)
    fig = go.Figure()
    fig.add_scatter(x=np.r_[grid, grid[::-1]], y=np.r_[sw["pi_high"], sw["pi_low"][::-1]], fill="toself",
                    fillcolor="rgba(52,152,219,0.18)", line=dict(width=0), name="預測區間")
    fig.add_scatter(x=np.r_[grid, grid[::-1]], y=np.r_[sw["ci_high"], sw["ci_low"][::-1]], fill="toself",
                    fillcolor="rgba(52,152,219,0.45)", line=dict(width=0), name="信賴區間")
    fig.add_scatter(x=grid, y=sw["pred"], mode="lines", line=dict(color="#1f4e79"), name="預測值")
    fig.update_layout(xaxis_title=feat, yaxis_title=target, height=420)
    st.plotly_chart(fig, width="stretch")

    st.download_button(
        "下載測試集預測結果（CSV）",
        test_iv.join(prep.y_test.rename("actual")).to_csv().encode("utf-8-sig"),
        file_name="predictions.csv",
        mime="text/csv",
    )
