"""RFM scoring, named segments and k-means clustering. Pure functions, unit-tested."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["recency_days", "frequency", "monetary"]

# Classic RFM grid on (recency score, frequency score). Order matters: first match wins.
SEGMENT_RULES: list[tuple[str, str]] = [
    ("Champions",           r"^5[45]$"),
    ("Loyal",               r"^[34][45]$"),
    ("Potential loyalists", r"^[45][23]$"),
    ("New customers",       r"^51$"),
    ("Promising",           r"^41$"),
    ("Need attention",      r"^33$"),
    ("About to sleep",      r"^3[12]$"),
    ("Can't lose",          r"^[12]5$"),
    ("At risk",             r"^[12][34]$"),
    ("Hibernating",         r"^[12][12]$"),
]
SEGMENT_ORDER = [name for name, _ in SEGMENT_RULES]

ACTIONS = {
    "Champions":           "Reward and protect: early access, loyalty perks, ask for referrals. No discounts needed.",
    "Loyal":               "Upsell and cross-sell; invite to the loyalty programme.",
    "Potential loyalists": "Push the next order: bundles, free-shipping threshold, membership offer.",
    "New customers":       "Onboarding series and a second-order incentive within 30 days.",
    "Promising":           "Introduce best-sellers; small time-limited offer to build the habit.",
    "Need attention":      "Personalised recommendations based on past orders; limited-time offer.",
    "About to sleep":      "Reactivation email with popular items; low-cost incentive.",
    "Can't lose":          "Personal outreach from an account manager; win-back offer. Highest-value lapsed customers.",
    "At risk":             "Win-back campaign with a meaningful offer; survey why they stopped.",
    "Hibernating":         "Low-cost automated reminders only; drop from paid channels if no response.",
}


def score(values: pd.Series, higher_is_better: bool = True, bins: int = 5) -> pd.Series:
    """1-5 score from percentile rank. Ties get the same score (method='average'),
    so e.g. every one-order customer gets the same frequency score."""
    s = values if higher_is_better else -values
    pct = s.rank(pct=True, method="average")
    return np.ceil(pct * bins).clip(1, bins).astype(int)


def name_segment(r: int, f: int) -> str:
    import re
    key = f"{r}{f}"
    for name, pattern in SEGMENT_RULES:
        if re.match(pattern, key):
            return name
    raise ValueError(f"no segment for R={r}, F={f}")


def add_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["R"] = score(out["recency_days"], higher_is_better=False)
    out["F"] = score(out["frequency"])
    out["M"] = score(out["monetary"])
    out["rfm_code"] = out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
    out["segment"] = [name_segment(r, f) for r, f in zip(out["R"], out["F"])]
    return out


def kmeans_features(df: pd.DataFrame) -> np.ndarray:
    """log1p tames the long right tails (a few wholesalers spend 100x the median);
    standardising puts the three features on the same scale."""
    return StandardScaler().fit_transform(np.log1p(df[FEATURES].to_numpy(dtype=float)))


def choose_k(X: np.ndarray, ks=range(2, 9), seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def order_clusters(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 1..k from highest to lowest median spend, so labels are stable and readable."""
    med = pd.Series(df["monetary"].to_numpy()).groupby(labels).median().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(med.index, start=1)}
    return np.array([mapping[l] for l in labels])


def eta_squared(groups: pd.Series, y: pd.Series) -> float:
    """Share of the variance in y explained by group membership (one-way ANOVA effect size)."""
    grand = y.mean()
    between = y.groupby(groups).apply(lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    total = ((y - grand) ** 2).sum()
    return float(between / total)
