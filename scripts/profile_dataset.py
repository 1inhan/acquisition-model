from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "startup data.csv"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    profile = {
        "dataset": DATA_PATH.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "target_columns": {
            "status": df["status"].value_counts(dropna=False).to_dict(),
            "labels": df["labels"].value_counts(dropna=False).to_dict(),
        },
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "numeric_describe": df.select_dtypes(include="number").describe().round(4).to_dict(),
        "categorical_cardinality": {
            col: int(df[col].nunique(dropna=True))
            for col in df.select_dtypes(exclude="number").columns
        },
    }

    (OUTPUT_DIR / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(100 * df[col].isna().mean(), 2),
                "unique_values": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        ]
    ).to_csv(OUTPUT_DIR / "column_profile.csv", index=False)

    print("Profiled:", DATA_PATH.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Wrote dataset_profile.json and column_profile.csv")


if __name__ == "__main__":
    main()
