"""Score, segment and backtest customers. Writes every number quoted in the README to reports/.

Run from the project root, after build.py:  python src/segment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import BACKTEST_SNAPSHOT, CURRENT_SNAPSHOT, REPORTS, ROOT  # noqa: E402
from rfm import (ACTIONS, SEGMENT_ORDER, add_rfm_scores, choose_k, eta_squared,  # noqa: E402
                 fit_kmeans, kmeans_features, order_clusters)

K = 4   # see README: k=2 only splits active/inactive; k=3 and k=4 tie on silhouette,
        # and k=4 separates recent low spenders from lapsed one-time buyers.
CLUSTER_NAMES = {1: "VIP regulars", 2: "Steady mid-value", 3: "Recent light buyers", 4: "Lapsed one-timers"}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / f"rfm_{name}.parquet")


def segment(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rfm_scores(df)
    X = kmeans_features(df)
    df["cluster"] = order_clusters(df, fit_kmeans(X, K).labels_)
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)
    return df


def summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by).agg(customers=("customer_id", "size"),
                           median_recency_days=("recency_days", "median"),
                           median_orders=("frequency", "median"),
                           median_spend=("monetary", "median"),
                           total_spend=("monetary", "sum"))
    g["share_customers"] = g["customers"] / g["customers"].sum()
    g["share_spend"] = g["total_spend"] / g["total_spend"].sum()
    return g


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(bought_again=df["next_orders"] > 0)
    g = df.groupby("segment").agg(customers=("customer_id", "size"),
                                  bought_again=("bought_again", "mean"),
                                  next_year_revenue_per_customer=("next_revenue", "mean"),
                                  next_year_revenue=("next_revenue", "sum"))
    g["share_next_year_revenue"] = g["next_year_revenue"] / g["next_year_revenue"].sum()
    return g.reindex(SEGMENT_ORDER)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    results: dict = {"k": K}

    # ---------- Backtest: segment at 2010-12-10, check the next 365 days ----------
    bt = segment(load("backtest"))
    bt_seg = backtest(bt)
    bt_seg.to_csv(REPORTS / "backtest_by_segment.csv")
    bt_clu = bt.assign(bought_again=bt.next_orders > 0).groupby("cluster_name").agg(
        customers=("customer_id", "size"), bought_again=("bought_again", "mean"),
        next_year_revenue_per_customer=("next_revenue", "mean"))
    bt_clu.to_csv(REPORTS / "backtest_by_cluster.csv")

    y = np.log1p(bt["next_revenue"].clip(lower=0))
    results["backtest"] = {
        "snapshot": BACKTEST_SNAPSHOT, "customers": len(bt),
        "bought_again_overall": float((bt.next_orders > 0).mean()),
        "eta2_named_segments": eta_squared(bt["segment"], y),
        "eta2_kmeans": eta_squared(bt["cluster"], y),
        "eta2_full_rfm_code": eta_squared(bt["rfm_code"], y),
        "n_rfm_codes": int(bt["rfm_code"].nunique()),
    }

    # ---------- Current segmentation: 2011-12-10 ----------
    cur = segment(load("current"))
    X = kmeans_features(cur)
    ksel = choose_k(X)
    ksel.to_csv(REPORTS / "k_selection.csv", index=False)
    labels = [fit_kmeans(X, K, seed=s).labels_ for s in range(10)]
    ari = [adjusted_rand_score(labels[0], l) for l in labels[1:]]

    seg = summary(cur, "segment").reindex(SEGMENT_ORDER)
    seg["action"] = seg.index.map(ACTIONS)
    seg.to_csv(REPORTS / "segments_current.csv")
    clu = summary(cur, "cluster_name").reindex(list(CLUSTER_NAMES.values()))
    clu.to_csv(REPORTS / "clusters_current.csv")
    pd.crosstab(cur["segment"], cur["cluster_name"]).reindex(SEGMENT_ORDER)[
        list(CLUSTER_NAMES.values())].to_csv(REPORTS / "segment_vs_cluster.csv")

    cur[["customer_id", "country", "recency_days", "frequency", "monetary", "R", "F", "M",
         "rfm_code", "segment", "cluster_name"]].sort_values("monetary", ascending=False).to_csv(
        REPORTS / "customer_segments.csv", index=False)

    spend = cur["monetary"].sort_values(ascending=False)
    results["current"] = {
        "snapshot": CURRENT_SNAPSHOT, "customers": len(cur),
        "total_spend": float(spend.sum()),
        "top20pct_share_of_spend": float(spend.head(int(len(spend) * 0.2)).sum() / spend.sum()),
        "one_order_share": float((cur.frequency == 1).mean()),
        "uk_share": float((cur.country == "United Kingdom").mean()),
        "median_spend": float(cur.monetary.median()),
        "silhouette": ksel.set_index("k")["silhouette"].round(3).to_dict(),
        "kmeans_ari_across_10_seeds_min": float(min(ari)),
        "f_scores_present": sorted(cur["F"].unique().tolist()),
    }
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))

    pd.set_option("display.width", 200)
    print(json.dumps(results, indent=2, default=float))
    print(bt_seg.round(3))
    print(bt_clu.round(3))
    print(seg.drop(columns="action").round(3))
    print(clu.round(3))
    print(pd.read_csv(REPORTS / "segment_vs_cluster.csv").to_string(index=False))


if __name__ == "__main__":
    main()
