"""Multiple linear regression pipeline: data preparation, feature selection,
OLS fitting with confidence / prediction intervals, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_CATEGORY_LEVELS = 20

PROBLEMS = {
    "life": {
        "label": "WHO 平均壽命預測（Life Expectancy）",
        "path": DATA_DIR / "life_expectancy.csv",
        "kaggle": "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who",
        "target": "Life expectancy",
        "drop": ["Country"],
        "group": "Country",
        "unit": "歲",
        "business": (
            "世界衛生組織（WHO）與各國衛生部門需要知道哪些因素最能解釋一國的平均壽命，"
            "以便把有限的預算投入在疫苗、教育或醫療支出等最有效的項目上。"
            "本題以 2000–2015 年 193 國的健康、經濟與社會指標，預測國家年度平均壽命（歲）。"
        ),
        "success": "測試集 RMSE < 4 歲、R² > 0.8，且 95% 預測區間的實際覆蓋率接近 95%。",
    },
    "flood": {
        "label": "洪水發生機率預測（Flood Prediction）",
        "path": DATA_DIR / "flood.csv",
        "kaggle": "https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset",
        "target": "FloodProbability",
        "drop": [],
        "group": None,
        "unit": "",
        "business": (
            "防災單位需要依據季風強度、排水系統、都市化、森林砍伐等 20 項環境與治理指標（1–10 分），"
            "估計一個地區的洪水發生機率，作為排定防洪工程與疏散演練優先順序的依據。"
        ),
        "success": "測試集 R² > 0.8，並找出對洪水機率影響最大的少數可治理因子。",
    },
}

SELECTION_METHODS = {
    "Forward Stepwise（逐步前進選擇）": "forward",
    "RFE（遞迴特徵消除）": "rfe",
    "Lasso（L1 正則化）": "lasso",
    "SelectKBest（F 檢定）": "kbest",
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated / trailing whitespace in column names."""
    df = df.copy()
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df


def load_dataset(source: str | Path) -> pd.DataFrame:
    """Load a CSV from a local path or URL (http / https)."""
    df = pd.read_csv(source, sep=None, engine="python")
    df = clean_columns(df)
    if df.shape[1] < 2:
        raise ValueError("資料集至少需要兩個欄位（特徵 + 目標）。")
    return df


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    log: list[str] = field(default_factory=list)


def prepare_data(
    df: pd.DataFrame,
    target: str,
    drop: list[str] | None = None,
    group: str | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> PreparedData:
    """Split first, then impute / encode using training statistics only."""
    log = []
    drop = [c for c in (drop or []) if c in df.columns and c != target]
    data = df.copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    n_missing_y = int(data[target].isna().sum())
    data = data.dropna(subset=[target]).reset_index(drop=True)
    if n_missing_y:
        log.append(f"刪除目標值缺失的 {n_missing_y} 筆資料。")

    groups = data[group] if group and group in data.columns else None
    features = data.drop(columns=[target, *drop])
    if drop:
        log.append(f"排除識別欄位：{', '.join(drop)}。")

    cat_cols = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    too_many = [c for c in cat_cols if features[c].nunique() > MAX_CATEGORY_LEVELS]
    if too_many:
        features = features.drop(columns=too_many)
        log.append(f"類別數超過 {MAX_CATEGORY_LEVELS} 的欄位不適合直接編碼，已排除：{', '.join(too_many)}。")
    cat_cols = [c for c in cat_cols if c not in too_many]

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(features, groups=groups))
        log.append(f"依「{group}」分組切分（同一組只出現在訓練或測試其中一邊），避免資料洩漏。")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(features)), test_size=test_size, random_state=seed
        )
    X_train, X_test = features.iloc[train_idx].copy(), features.iloc[test_idx].copy()
    y_train, y_test = data[target].iloc[train_idx], data[target].iloc[test_idx]
    log.append(f"切分訓練集 {len(X_train)} 筆、測試集 {len(X_test)} 筆（test_size={test_size}）。")

    num_cols = [c for c in X_train.columns if c not in cat_cols]
    medians = X_train[num_cols].median()
    n_imputed = int(X_train[num_cols].isna().sum().sum() + X_test[num_cols].isna().sum().sum())
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_test[num_cols] = X_test[num_cols].fillna(medians)
    if n_imputed:
        log.append(f"以訓練集中位數填補 {n_imputed} 個數值缺失值。")

    if cat_cols:
        for c in cat_cols:
            mode = X_train[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "missing"
            X_train[c] = X_train[c].fillna(fill)
            X_test[c] = X_test[c].fillna(fill)
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=float)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=float)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        log.append(f"類別欄位 One-Hot 編碼（drop_first 避免虛擬變數陷阱）：{', '.join(cat_cols)}。")

    constant = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    if constant:
        X_train, X_test = X_train.drop(columns=constant), X_test.drop(columns=constant)
        log.append(f"移除常數欄位：{', '.join(constant)}。")
    if X_train.shape[1] == 0:
        raise ValueError("前處理後沒有可用的特徵欄位。")

    return PreparedData(
        X_train.astype(float),
        X_test.astype(float),
        y_train.astype(float),
        y_test.astype(float),
        log,
    )


def _cv_r2(X: pd.DataFrame, y: pd.Series, cv: KFold) -> float:
    return float(cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2").mean())


def rank_features(X: pd.DataFrame, y: pd.Series, method: str, seed: int = 42) -> list[str]:
    """Return every feature ordered from most to least useful for the given method."""
    cols = list(X.columns)
    if method == "kbest":
        scores, _ = f_regression(X, y)
        return [cols[i] for i in np.argsort(-np.nan_to_num(scores))]
    if method == "rfe":
        Xs = StandardScaler().fit_transform(X)
        rfe = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, y)
        return [cols[i] for i in np.argsort(rfe.ranking_)]
    if method == "lasso":
        Xs = StandardScaler().fit_transform(X)
        alphas = LassoCV(cv=5, random_state=seed, max_iter=20000).fit(Xs, y).alphas_
        # Order by the strongest penalty at which each feature first becomes non-zero,
        # then by coefficient size at the CV-optimal alpha.
        _, coefs, _ = lasso_path(Xs, y, alphas=alphas, max_iter=20000)
        entry = [int(np.argmax(np.abs(c) > 1e-10)) if np.any(np.abs(c) > 1e-10) else len(alphas) for c in coefs]
        final = np.abs(coefs[:, -1])
        return [cols[i] for i in sorted(range(len(cols)), key=lambda i: (entry[i], -final[i]))]
    if method == "forward":
        cv = KFold(5, shuffle=True, random_state=seed)
        chosen, remaining = [], cols.copy()
        while remaining:
            best = max(remaining, key=lambda c: _cv_r2(X[chosen + [c]], y, cv))
            chosen.append(best)
            remaining.remove(best)
        return chosen
    raise ValueError(f"未知的特徵選擇方法：{method}")


def k_curve(X: pd.DataFrame, y: pd.Series, ranking: list[str], seed: int = 42) -> pd.DataFrame:
    """5-fold CV RMSE / R² / adjusted R² for the top-k features, k = 1..p."""
    cv = KFold(5, shuffle=True, random_state=seed)
    rows = []
    n = len(y)
    for k in range(1, len(ranking) + 1):
        Xk = X[ranking[:k]]
        r2 = float(cross_val_score(LinearRegression(), Xk, y, cv=cv, scoring="r2").mean())
        rmse = float(-cross_val_score(LinearRegression(), Xk, y, cv=cv, scoring="neg_root_mean_squared_error").mean())
        adj = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
        rows.append({"k": k, "CV_R2": r2, "CV_Adj_R2": adj, "CV_RMSE": rmse, "added": ranking[k - 1]})
    return pd.DataFrame(rows)


def suggest_k(curve: pd.DataFrame, tolerance: float = 0.01) -> int:
    """Smallest k whose CV RMSE is within `tolerance` (relative) of the best RMSE."""
    best = curve["CV_RMSE"].min()
    ok = curve[curve["CV_RMSE"] <= best * (1 + tolerance)]
    return int(ok["k"].min())


def fit_ols(X: pd.DataFrame, y: pd.Series):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()


def intervals(result, X: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Point prediction with confidence interval (mean) and prediction interval (new obs)."""
    frame = result.get_prediction(sm.add_constant(X, has_constant="add")).summary_frame(alpha=alpha)
    frame.index = X.index
    return frame.rename(
        columns={
            "mean": "pred",
            "mean_ci_lower": "ci_low",
            "mean_ci_upper": "ci_high",
            "obs_ci_lower": "pi_low",
            "obs_ci_upper": "pi_high",
        }
    )[["pred", "ci_low", "ci_high", "pi_low", "pi_high"]]


def metrics(y_true, y_pred, k: int) -> dict[str, float]:
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1),
    }


def compare_models(prep: PreparedData, selected: list[str], seed: int = 42) -> pd.DataFrame:
    """Test-set comparison: OLS on all features vs. selected features, Ridge and Lasso."""
    all_cols = list(prep.X_train.columns)
    candidates = {
        f"OLS 全部特徵（{len(all_cols)}）": (LinearRegression(), all_cols),
        f"OLS 選取特徵（{len(selected)}）": (LinearRegression(), selected),
        "Ridge（全部特徵，CV α）": (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))), all_cols),
        "Lasso（全部特徵，CV α）": (make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=seed, max_iter=20000)), all_cols),
    }
    rows = []
    for name, (model, cols) in candidates.items():
        model.fit(prep.X_train[cols], prep.y_train)
        pred = model.predict(prep.X_test[cols])
        rows.append({"模型": name, "特徵數": len(cols), **metrics(prep.y_test, pred, len(cols))})
    return pd.DataFrame(rows)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X, has_constant="add").values
    vifs = [variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])]
    return pd.DataFrame({"feature": X.columns, "VIF": vifs}).sort_values("VIF", ascending=False)


def coefficient_table(result, alpha: float = 0.05) -> pd.DataFrame:
    ci = result.conf_int(alpha=alpha)
    table = pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "t": result.tvalues,
            "p_value": result.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    return table


def assumption_tests(result) -> pd.DataFrame:
    """Durbin–Watson, Jarque–Bera (normality), Breusch–Pagan (homoscedasticity)."""
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    resid = result.resid
    jb, jb_p, _, _ = jarque_bera(resid)
    bp, bp_p, _, _ = het_breuschpagan(resid, result.model.exog)
    dw = durbin_watson(resid)
    return pd.DataFrame(
        [
            {"假設": "誤差獨立（Durbin–Watson）", "統計量": dw, "p 值": np.nan, "判讀": "接近 2 表示無自相關" if 1.5 < dw < 2.5 else "可能存在自相關"},
            {"假設": "誤差常態（Jarque–Bera）", "統計量": jb, "p 值": jb_p, "判讀": "無法拒絕常態" if jb_p > 0.05 else "殘差偏離常態"},
            {"假設": "變異數同質（Breusch–Pagan）", "統計量": bp, "p 值": bp_p, "判讀": "無法拒絕同質" if bp_p > 0.05 else "存在異質變異"},
        ]
    )
