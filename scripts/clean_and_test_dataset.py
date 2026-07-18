from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "startup data.csv"
OUTPUT_DIR = ROOT / "outputs"

QUANTITATIVE_FEATURES = [
    "age_first_funding_year",
    "age_last_funding_year",
    "relationships",
    "funding_rounds",
    "funding_total_usd",
    "milestones",
    "age_first_milestone_year",
    "age_last_milestone_year",
    "avg_participants",
]

CATEGORICAL_FEATURES = [
    "state",
    "industry_type",
    "has_VC",
    "has_angel",
    "has_roundA",
    "has_roundB",
    "has_roundC",
    "has_roundD",
]

TARGET = "is_acquired"


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False).statistic
    n = table.to_numpy().sum()
    k = min(table.shape)
    return float(np.sqrt(chi2 / (n * (k - 1)))) if n and k > 1 else np.nan


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.rename(columns={"state_code": "state", "category_code": "industry_type"})
    df[TARGET] = (df["status"].astype(str).str.lower() == "acquired").astype(int)

    for col in QUANTITATIVE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["state"] = df["state"].fillna("Unknown").astype(str)
    df["industry_type"] = df["industry_type"].fillna("Unknown").astype(str)
    df["funding_total_usd_log10"] = np.log10(df["funding_total_usd"].clip(lower=0) + 1)
    df["milestone_age_span"] = df["age_last_milestone_year"] - df["age_first_milestone_year"]
    df["funding_age_span"] = df["age_last_funding_year"] - df["age_first_funding_year"]
    df["has_milestone_age"] = df["age_first_milestone_year"].notna().astype(int)

    selected = [
        "id",
        "name",
        "status",
        TARGET,
        *QUANTITATIVE_FEATURES,
        *CATEGORICAL_FEATURES,
        "funding_total_usd_log10",
        "milestone_age_span",
        "funding_age_span",
        "has_milestone_age",
    ]
    return df[selected].copy()


def data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("row_count_positive", len(df) > 0, len(df), "Dataset has rows."),
        ("id_unique", df["id"].is_unique, int(df["id"].nunique()), "Startup IDs should be unique."),
        ("target_complete", df[TARGET].notna().all(), int(df[TARGET].isna().sum()), "Target should be complete."),
        ("target_binary", set(df[TARGET].dropna().unique()) <= {0, 1}, sorted(df[TARGET].unique()), "Target should be binary."),
        (
            "target_has_both_classes",
            df[TARGET].nunique() == 2,
            df[TARGET].value_counts().to_dict(),
            "Both acquired and closed classes should exist.",
        ),
        (
            "funding_total_nonnegative",
            (df["funding_total_usd"].dropna() >= 0).all(),
            float(df["funding_total_usd"].min()),
            "Funding totals should be nonnegative.",
        ),
        (
            "quantitative_missing_under_20pct",
            df[QUANTITATIVE_FEATURES].isna().mean().max() < 0.20,
            round(100 * df[QUANTITATIVE_FEATURES].isna().mean().max(), 2),
            "No quantitative feature should exceed 20% missingness.",
        ),
        (
            "categorical_complete",
            df[CATEGORICAL_FEATURES].notna().all().all(),
            int(df[CATEGORICAL_FEATURES].isna().sum().sum()),
            "Categorical model inputs should be complete after cleaning.",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "observed": observed, "note": note}
            for name, passed, observed, note in checks
        ]
    )


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in QUANTITATIVE_FEATURES:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "type": "quantitative",
                "missing_pct": round(100 * s.isna().mean(), 2),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "unique_values": int(s.nunique(dropna=True)),
            }
        )
    for col in CATEGORICAL_FEATURES:
        s = df[col].astype(str)
        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "missing_pct": round(100 * df[col].isna().mean(), 2),
                "mean": "",
                "median": "",
                "std": "",
                "min": "",
                "max": "",
                "unique_values": int(s.nunique(dropna=True)),
                "top_value": s.value_counts().idxmax(),
                "top_share_pct": round(100 * s.value_counts(normalize=True).max(), 2),
            }
        )
    return pd.DataFrame(rows)


def numeric_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    acquired = df[df[TARGET] == 1]
    closed = df[df[TARGET] == 0]
    for col in QUANTITATIVE_FEATURES:
        a = acquired[col].dropna()
        c = closed[col].dropna()
        statistic, p_value = stats.mannwhitneyu(a, c, alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "test": "Mann-Whitney U",
                "acquired_median": round(float(a.median()), 4),
                "closed_median": round(float(c.median()), 4),
                "median_difference": round(float(a.median() - c.median()), 4),
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_FEATURES:
        table = pd.crosstab(df[col], df[TARGET])
        chi = stats.chi2_contingency(table, correction=False)
        rows.append(
            {
                "feature": col,
                "test": "chi-square",
                "levels": int(df[col].nunique()),
                "chi_square": round(float(chi.statistic), 4),
                "p_value": round(float(chi.pvalue), 6),
                "cramers_v": round(cramers_v(table), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def status_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for status_value, group in df.groupby("status"):
        row = {
            "status": status_value,
            "rows": int(len(group)),
            "share_pct": round(100 * len(group) / len(df), 2),
        }
        for col in QUANTITATIVE_FEATURES:
            row[f"{col}_median"] = round(float(group[col].median()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean_dataset(raw)

    outputs = {
        "cleaned_startups.csv": cleaned,
        "data_quality_checks.csv": data_quality_checks(cleaned),
        "feature_summary.csv": feature_summary(cleaned),
        "target_distribution.csv": cleaned["status"].value_counts().rename_axis("status").reset_index(name="rows"),
        "numeric_tests_by_status.csv": numeric_tests(cleaned),
        "categorical_tests_by_status.csv": categorical_tests(cleaned),
        "status_group_summary.csv": status_group_summary(cleaned),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    profile = {
        "raw_file": RAW_PATH.name,
        "rows": int(len(cleaned)),
        "columns": int(len(cleaned.columns)),
        "target": TARGET,
        "target_distribution": cleaned["status"].value_counts().to_dict(),
        "quantitative_features": QUANTITATIVE_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_leakage_columns": ["labels", "status"],
    }
    (OUTPUT_DIR / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print("Cleaned startup rows:", len(cleaned))
    print("Target distribution:", cleaned["status"].value_counts().to_dict())
    print("Wrote startup cleaning/test outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
