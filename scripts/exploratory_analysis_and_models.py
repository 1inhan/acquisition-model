from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from clean_and_test_dataset import (
    CATEGORICAL_FEATURES,
    OUTPUT_DIR,
    QUANTITATIVE_FEATURES,
    RAW_PATH,
    TARGET,
    clean_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), QUANTITATIVE_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def model_specs() -> dict[str, Pipeline]:
    return {
        "gaussian_naive_bayes": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", GaussianNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "knn_7": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "svm_linear": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "bagging_trees": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    BaggingClassifier(
                        estimator=DecisionTreeClassifier(
                            max_depth=6,
                            min_samples_leaf=10,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "adaboost": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.01,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_holdout(models: dict[str, Pipeline], x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    confusion_rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": name,
                "split": "stratified_holdout",
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "accuracy": round(accuracy_score(y_test, pred), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
                "precision": round(precision_score(y_test, pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, prob), 4),
                "pr_auc": round(average_precision_score(y_test, prob), 4),
            }
        )
        confusion_rows.append(
            {
                "model": name,
                "true_closed_pred_closed": int(tn),
                "true_closed_pred_acquired": int(fp),
                "true_acquired_pred_closed": int(fn),
                "true_acquired_pred_acquired": int(tp),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def evaluate_cv(models: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, x, y, scoring=scoring, cv=cv, n_jobs=1)
        row = {"model": name, "split": "5_fold_stratified_cv"}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_std"] = round(float(values.std()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_rows(y_test: pd.Series) -> pd.DataFrame:
    majority = int(y_test.mode().iloc[0])
    pred = np.repeat(majority, len(y_test))
    prob = np.repeat(float(majority), len(y_test))
    return pd.DataFrame(
        [
            {
                "model": "majority_baseline",
                "split": "stratified_holdout",
                "train_rows": "",
                "test_rows": int(len(y_test)),
                "accuracy": round(accuracy_score(y_test, pred), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
                "precision": round(precision_score(y_test, pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, prob), 4),
                "pr_auc": round(average_precision_score(y_test, prob), 4),
            }
        ]
    )


def tree_feature_importance(model: Pipeline, model_name: str) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame()
    return (
        pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(30)
    )


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = estimator.coef_[0]
    frame = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).head(30)


def decision_tree_rules(model: Pipeline) -> str:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    return export_text(estimator, feature_names=feature_names, decimals=3, spacing=2)


def decision_tree_leaf_summary(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    transformed = model.named_steps["preprocess"].transform(x_test)
    estimator = model.named_steps["model"]
    leaves = estimator.apply(transformed)
    pred = estimator.predict(transformed)
    prob = estimator.predict_proba(transformed)[:, 1]
    frame = pd.DataFrame(
        {
            "leaf_id": leaves,
            "actual": y_test.to_numpy(),
            "predicted": pred,
            "prob_acquired": prob,
        }
    )
    summary = (
        frame.groupby("leaf_id")
        .agg(
            rows=("actual", "size"),
            actual_acquired_rate=("actual", "mean"),
            predicted_acquired_rate=("predicted", "mean"),
            avg_predicted_probability=("prob_acquired", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "avg_predicted_probability"], ascending=[False, False])
    )
    return summary.round(4)


def write_interpretability_report(
    holdout: pd.DataFrame,
    feature_importance: pd.DataFrame,
    logistic: pd.DataFrame,
    leaf_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Model Interpretability Report",
        "",
        "## What can be inspected directly",
        "",
        "- Decision Tree: explicit if/else rules are exported in `decision_tree_rules.txt`.",
        "- Logistic Regression: signed coefficients are exported in `logistic_coefficients.csv`.",
        "- Tree ensembles: aggregate feature importances are exported in `feature_importance.csv`.",
        "- KNN, RBF SVM, Naive Bayes, and boosting models are less directly readable from simple rules.",
        "",
        "## Model performance context",
        "",
        markdown_table(holdout.sort_values("roc_auc", ascending=False)),
        "",
        "## Top feature importances",
        "",
        markdown_table(feature_importance.head(25)),
        "",
        "## Top logistic coefficients",
        "",
        markdown_table(logistic.head(25)),
        "",
        "## Decision tree leaf summary",
        "",
        markdown_table(leaf_summary),
        "",
        "## RNN note",
        "",
        "No RNN has been trained yet. The current dataset is row-level tabular data, not a true sequence dataset. An RNN would make more sense after converting each startup into a timestamped funding/milestone event sequence.",
    ]
    (OUTPUT_DIR / "model_interpretability_report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    display = display.astype(str)
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in display.values.tolist())
    return "\n".join(lines)


def write_report(holdout: pd.DataFrame, cv: pd.DataFrame, feature_importance: pd.DataFrame, logistic: pd.DataFrame, target_distribution: pd.Series) -> None:
    best = holdout.sort_values("roc_auc", ascending=False)
    lines = [
        "# Startup Acquisition Baseline Model Report",
        "",
        "## Target Distribution",
        "",
        markdown_table(target_distribution.rename_axis("status").reset_index(name="rows")),
        "",
        "## Holdout Results",
        "",
        markdown_table(best),
        "",
        "## Cross-Validation Results",
        "",
        markdown_table(cv.sort_values("roc_auc_mean", ascending=False)),
        "",
        "## Top Tree-Based Feature Importances",
        "",
        markdown_table(feature_importance.head(30)),
        "",
        "## Top Logistic Regression Coefficients",
        "",
        markdown_table(logistic.head(30)),
        "",
        "## Notes",
        "",
        "- Target is `is_acquired`, derived from `status == acquired`.",
        "- `labels` and `status` are excluded from model inputs to avoid leakage.",
        "- Metrics include accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.",
    ]
    (OUTPUT_DIR / "exploratory_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    df = clean_dataset(raw)
    x = df[QUANTITATIVE_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = model_specs()
    holdout, confusion = evaluate_holdout(models, x_train, x_test, y_train, y_test)
    holdout = pd.concat([baseline_rows(y_test), holdout], ignore_index=True)
    cv = evaluate_cv(models, x, y)

    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model

    importances = pd.concat(
        [
            tree_feature_importance(fitted_models[name], name)
            for name in [
                "decision_tree",
                "bagging_trees",
                "random_forest",
                "extra_trees",
                "adaboost",
                "gradient_boosting",
                "hist_gradient_boosting",
            ]
        ],
        ignore_index=True,
    )
    logistic = logistic_coefficients(fitted_models["logistic_regression"])
    rules = decision_tree_rules(fitted_models["decision_tree"])
    leaf_summary = decision_tree_leaf_summary(fitted_models["decision_tree"], x_test, y_test)

    holdout.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False)
    cv.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)
    confusion.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
    importances.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    logistic.to_csv(OUTPUT_DIR / "logistic_coefficients.csv", index=False)
    leaf_summary.to_csv(OUTPUT_DIR / "decision_tree_leaf_summary.csv", index=False)
    (OUTPUT_DIR / "decision_tree_rules.txt").write_text(rules, encoding="utf-8")
    write_report(holdout, cv, importances, logistic, df["status"].value_counts())
    write_interpretability_report(holdout, importances, logistic, leaf_summary)

    print("Wrote startup model outputs to:", OUTPUT_DIR)
    print("Best holdout model by ROC-AUC:", holdout.sort_values("roc_auc", ascending=False).iloc[0]["model"])


if __name__ == "__main__":
    main()
