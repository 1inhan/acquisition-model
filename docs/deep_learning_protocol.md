# Modeling Protocol for Startup Acquisition Prediction

## Current dataset

The active dataset is `startup data.csv`, a structured startup-level dataset with funding, milestone, geography, industry, investor, and funding-round indicators.

The target is acquisition outcome:

- `status = acquired` becomes `is_acquired = 1`
- `status = closed` becomes `is_acquired = 0`

Do not use `labels` as a feature. It duplicates the target.

## Current feature set

Quantitative features:

- `age_first_funding_year`
- `age_last_funding_year`
- `relationships`
- `funding_rounds`
- `funding_total_usd`
- `milestones`
- `age_first_milestone_year`
- `age_last_milestone_year`
- `avg_participants`

Categorical features:

- `state`
- `industry_type`
- `has_VC`
- `has_angel`
- `has_roundA`
- `has_roundB`
- `has_roundC`
- `has_roundD`

## Baseline modeling protocol

The current baseline is intentionally classical:

1. Clean and validate the dataset.
2. Summarize feature distributions and missingness.
3. Run acquired-vs-closed statistical tests.
4. Train transparent baseline models.
5. Compare against a majority-class baseline.

Current baseline models:

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Linear and RBF SVM
- Decision Tree
- Bagging Trees
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- Histogram Gradient Boosting

Metrics:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

## Why this is better than the old acquisition-history framing

The previous acquisition-history CSV only listed observed acquisitions. It did not include comparable non-acquired companies, so it was weak for supervised prediction.

This startup dataset is much better for predictive modeling because it includes both acquired and closed startups, plus company-level predictors that exist before the outcome.

## Current findings

The strongest descriptive signals are:

- `relationships`
- `funding_total_usd`
- `milestones`
- `age_last_milestone_year`
- `avg_participants`
- later funding rounds, especially `has_roundB`, `has_roundC`, and `has_roundD`

Gradient Boosting is currently the strongest holdout model, with ROC-AUC around `0.8741` after removing the poorly defined `is_top500` feature.

## Next modeling direction

Good near-term improvements:

- Hyperparameter tuning.
- Calibration and threshold analysis.
- Permutation importance.
- SHAP explanations.
- Robust duplicate-ID handling.
- Sensitivity checks with and without geography/industry.

More advanced future direction:

- Time-aware split using `founded_at`, `first_funding_at`, and `last_funding_at`.
- Survival analysis or hazard models for time-to-acquisition.
- Sequence models over funding rounds and milestone events if timestamped event history can be expanded.
- Temporal graph models if investor-company relationships can be added.
