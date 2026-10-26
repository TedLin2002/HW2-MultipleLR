import numpy as np
import pandas as pd
import pytest

import mlr


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(rng.normal(size=(n, 15)), columns=[f"x{i}" for i in range(15)])
    df["group"] = rng.choice(list("abc"), n)
    df.loc[::17, "x3"] = np.nan
    df["y"] = 3 * df["x0"] - 2 * df["x1"] + (df["group"] == "b") * 1.5 + rng.normal(scale=0.5, size=n)
    return df


def test_prepare_has_no_missing_and_encodes(toy):
    prep = mlr.prepare_data(toy, "y")
    assert not prep.X_train.isna().any().any() and not prep.X_test.isna().any().any()
    assert list(prep.X_train.columns) == list(prep.X_test.columns)
    assert "group_b" in prep.X_train.columns


@pytest.mark.parametrize("method", mlr.SELECTION_METHODS.values())
def test_ranking_finds_true_signal(toy, method):
    prep = mlr.prepare_data(toy, "y")
    ranking = mlr.rank_features(prep.X_train, prep.y_train, method)
    assert set(ranking[:2]) == {"x0", "x1"}
    assert sorted(ranking) == sorted(prep.X_train.columns)


def test_prediction_interval_wider_than_confidence(toy):
    prep = mlr.prepare_data(toy, "y")
    cols = ["x0", "x1", "group_b"]
    res = mlr.fit_ols(prep.X_train[cols], prep.y_train)
    iv = mlr.intervals(res, prep.X_test[cols])
    assert ((iv.pi_high - iv.pi_low) > (iv.ci_high - iv.ci_low)).all()
    coverage = ((prep.y_test >= iv.pi_low) & (prep.y_test <= iv.pi_high)).mean()
    assert coverage > 0.85


def test_group_split_keeps_groups_apart():
    p = mlr.PROBLEMS["life"]
    df = mlr.load_dataset(p["path"])
    prep = mlr.prepare_data(df, p["target"], p["drop"], p["group"])
    countries = df.dropna(subset=[p["target"]]).reset_index(drop=True)["Country"]
    assert prep.X_train.shape[1] > 13
    assert set(countries[prep.X_train.index]).isdisjoint(countries[prep.X_test.index])
