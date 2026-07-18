# Startup Acquisition Prediction

This project will analyze startup history by examining features pertaining to funding, milestone, investor, geography, and industry features. As of currently, the sources dataset examines startups and whether they were successfully acquired. The current data is thus being tested to to establish whether the available company-level metrics contain useful signal for predicting whether a startup was `acquired` or `closed`. The project will incorporate time-based and sequencial modeling techniques, with the aim of predicting outcomes from the acquirer perspective, as is the current role of many VC analysts in the industry.

This README reflects the current dataset:

- `startup data.csv`
This is sourced from kaggle through the following link: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

Please thank (according to the Kaggle source):
- Ramkishan Panthena, for providing us this dataset. He is a Machine Learning Engineer at GMO.
- This dataset was used in data sprint #5 at DPhi.


## Dataset

Rows:

- 923 startups

Target:

- `status`
- Converted into binary target `is_acquired`
- `1 = acquired`
- `0 = closed`

Target distribution:

| Status | Rows | Share |
| --- | ---: | ---: |
| acquired | 597 | 64.7% |
| closed | 326 | 35.3% |

Important leakage note:

- `labels` duplicates the target and is excluded from model inputs.
- `status` is also excluded from model inputs.

## Model Inputs

The active model inputs are the features you listed.

Quantitative inputs:

- `age_first_funding_year`
- `age_last_funding_year`
- `relationships`
- `funding_rounds`
- `funding_total_usd`
- `milestones`
- `age_first_milestone_year`
- `age_last_milestone_year`
- `avg_participants`

Categorical inputs:

- `state`
- `industry_type`
- `has_VC`
- `has_angel`
- `has_roundA`
- `has_roundB`
- `has_roundC`
- `has_roundD`

Source column mapping:

- `state` comes from `state_code`
- `industry_type` comes from `category_code`

## Cleaning

Run:

```bash
.venv/bin/python scripts/clean_and_test_dataset.py
```

The cleaning script:

- Renames `state_code` to `state`.
- Renames `category_code` to `industry_type`.
- Creates `is_acquired` from `status`.
- Converts quantitative inputs to numeric.
- Converts binary categorical flags to integer 0/1.
- Fills missing `state` and `industry_type` as `Unknown`.
- Creates helper fields such as `funding_total_usd_log10`, but the baseline models use the original listed inputs.

Main cleaned output:

- `outputs/cleaned_startups.csv`

## Data Quality

Generated file:

- `outputs/data_quality_checks.csv`

Current checks:

| Check | Status | Observed |
| --- | --- | --- |
| Row count positive | pass | 923 |
| ID unique | fail | 922 unique IDs |
| Target complete | pass | 0 missing |
| Target binary | pass | 0/1 |
| Both target classes present | pass | 597 acquired, 326 closed |
| Funding total nonnegative | pass | minimum 11,000 |
| Quantitative missingness under 20% | pass | max 16.47% |
| Categorical complete | pass | 0 missing after cleaning |

Critical:

- There is one duplicate startup ID.

## Descriptive Metrics

Generated files:

- `outputs/feature_summary.csv`
- `outputs/status_group_summary.csv`
- `outputs/target_distribution.csv`

Here, we examine the biggest acquired-vs-closed differences are in company traction and funding maturity.

Median values by target:

| Feature | Acquired median | Closed median | Difference |
| --- | ---: | ---: | ---: |
| `relationships` | 7.0 | 3.0 | +4.0 |
| `funding_total_usd` | 12,700,000 | 5,000,000 | +7,700,000 |
| `milestones` | 2.0 | 1.0 | +1.0 |
| `age_first_milestone_year` | 3.0 | 1.2521 | +1.7479 |
| `age_last_milestone_year` | 5.0027 | 2.8219 | +2.1808 |
| `avg_participants` | 2.6667 | 2.0 | +0.6667 |
| `age_last_funding_year` | 3.7562 | 2.7192 | +1.0370 |
| `age_first_funding_year` | 1.4466 | 1.4384 | +0.0082 |

Interpretations (as expected from the startup perspective):

- Acquired startups tend to have more relationships.
- Acquired startups tend to raise more total funding.
- Acquired startups tend to hit more milestones.
- The first funding timing is not meaningfully different by itself.

## Statistical Tests

Generated files:

- `outputs/numeric_tests_by_status.csv`
- `outputs/categorical_tests_by_status.csv`

### Numeric Tests

Quantitative features are tested with Mann-Whitney U tests because the distributions are skewed and funding values are heavy-tailed.

Strongest numeric results:

| Feature | Test | p-value | Direction |
| --- | --- | ---: | --- |
| `relationships` | Mann-Whitney U | < 0.000001 | acquired higher |
| `funding_rounds` | Mann-Whitney U | < 0.000001 | acquired distribution higher |
| `funding_total_usd` | Mann-Whitney U | < 0.000001 | acquired higher |
| `milestones` | Mann-Whitney U | < 0.000001 | acquired higher |
| `age_first_milestone_year` | Mann-Whitney U | < 0.000001 | acquired higher |
| `age_last_milestone_year` | Mann-Whitney U | < 0.000001 | acquired higher |
| `avg_participants` | Mann-Whitney U | < 0.000001 | acquired higher |
| `age_last_funding_year` | Mann-Whitney U | 0.000013 | acquired higher |
| `age_first_funding_year` | Mann-Whitney U | 0.914823 | no useful difference |

### Categorical Tests

Categorical features are tested with chi-square tests.

Strongest categorical results:

| Feature | Cramer's V | p-value | Interpretation |
| --- | ---: | ---: | --- |
| `has_roundB` | 0.2083 | < 0.000001 | associated with acquisition status |
| `has_roundA` | 0.1843 | < 0.000001 | associated with acquisition status |
| `has_roundC` | 0.1659 | < 0.000001 | associated with acquisition status |
| `has_roundD` | 0.1399 | 0.000021 | associated with acquisition status |
| `state` | 0.2740 | 0.000329 | geography differs by outcome |
| `industry_type` | 0.2607 | 0.001940 | industry differs by outcome |
| `has_angel` | 0.0728 | 0.026901 | weak association |
| `has_VC` | 0.0565 | 0.085981 | not significant at 0.05 |

Interpretation:

- Funding rounds, geography, and industry type are relevant.
- `has_VC` alone is not a strong separator in this dataset.

## Baseline Machine Learning

Run:

```bash
.venv/bin/python scripts/exploratory_analysis_and_models.py
```

Generated files:

- `outputs/baseline_model_results.csv`
- `outputs/cross_validation_results.csv`
- `outputs/confusion_matrices.csv`
- `outputs/feature_importance.csv`
- `outputs/logistic_coefficients.csv`
- `outputs/exploratory_model_report.md`

Current Models:

- Majority baseline
- Gaussian Naive Bayes
- Logistic Regression
- K-Nearest Neighbors
- Linear SVM
- RBF SVM
- Decision Tree
- Bagging Trees
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- Histogram Gradient Boosting

Preprocessing:

- Median imputation for numeric features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical features.
- Standard scaling for Logistic Regression numeric inputs.

## Holdout Results

Holdout split:

- 75% train
- 25% test
- Stratified by target
- Random seed `42`

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.6450 | 0.5000 | 0.6450 | 1.0000 | 0.7842 | 0.5000 | 0.6450 |
| Gaussian Naive Bayes | 0.3550 | 0.4973 | 0.5000 | 0.0067 | 0.0132 | 0.3807 | 0.5981 |
| Logistic Regression | 0.6883 | 0.7008 | 0.8235 | 0.6577 | 0.7313 | 0.7822 | 0.8517 |
| K-Nearest Neighbors | 0.7489 | 0.6984 | 0.7692 | 0.8725 | 0.8176 | 0.7703 | 0.8360 |
| Linear SVM | 0.6840 | 0.6892 | 0.8065 | 0.6711 | 0.7326 | 0.7617 | 0.8380 |
| RBF SVM | 0.7229 | 0.7277 | 0.8346 | 0.7114 | 0.7681 | 0.8173 | 0.8701 |
| Decision Tree | 0.7056 | 0.6923 | 0.7914 | 0.7383 | 0.7639 | 0.7685 | 0.8293 |
| Bagging Trees | 0.7489 | 0.7450 | 0.8370 | 0.7584 | 0.7958 | 0.8429 | 0.8912 |
| Random Forest | 0.7489 | 0.7478 | 0.8421 | 0.7517 | 0.7943 | 0.8292 | 0.8843 |
| Extra Trees | 0.6407 | 0.6584 | 0.7946 | 0.5973 | 0.6820 | 0.7625 | 0.8348 |
| AdaBoost | 0.7879 | 0.7259 | 0.7778 | 0.9396 | 0.8511 | 0.8548 | 0.9062 |
| Gradient Boosting | 0.8095 | 0.7756 | 0.8261 | 0.8926 | 0.8581 | 0.8741 | 0.9201 |
| Histogram Gradient Boosting | 0.7662 | 0.7283 | 0.7950 | 0.8591 | 0.8258 | 0.8213 | 0.8750 |

Best holdout model:

- Gradient Boosting

Best holdout ROC-AUC:

- 0.8741

## Cross-Validation Results

Five-fold stratified CV:

| Model | Accuracy mean | Balanced accuracy mean | F1 mean | ROC-AUC mean | PR-AUC mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian Naive Bayes | 0.4300 | 0.5107 | 0.2304 | 0.4576 | 0.6481 |
| Logistic Regression | 0.7129 | 0.7147 | 0.7616 | 0.7818 | 0.8475 |
| K-Nearest Neighbors | 0.7432 | 0.6902 | 0.8142 | 0.7606 | 0.8192 |
| Linear SVM | 0.7075 | 0.7015 | 0.7615 | 0.7726 | 0.8442 |
| RBF SVM | 0.7367 | 0.7229 | 0.7911 | 0.8081 | 0.8559 |
| Decision Tree | 0.7108 | 0.6874 | 0.7723 | 0.7478 | 0.8108 |
| Bagging Trees | 0.7671 | 0.7380 | 0.8232 | 0.8018 | 0.8526 |
| Random Forest | 0.7432 | 0.7264 | 0.7978 | 0.8102 | 0.8640 |
| Extra Trees | 0.6944 | 0.6949 | 0.7446 | 0.7660 | 0.8292 |
| AdaBoost | 0.7725 | 0.7065 | 0.8410 | 0.8129 | 0.8628 |
| Gradient Boosting | 0.7845 | 0.7423 | 0.8417 | 0.8235 | 0.8648 |
| Histogram Gradient Boosting | 0.7660 | 0.7246 | 0.8274 | 0.7902 | 0.8391 |

Interpretation:

- Gradient Boosting is strongest overall on accuracy and F1.
- Random Forest is competitive and has strong ROC-AUC/PR-AUC.
- Logistic Regression performs reasonably, which suggests the signal is not purely nonlinear.
- Naive Bayes performs poorly, which suggests the Gaussian/independence assumptions do not fit this mixed startup dataset.
- KNN and SVM variants are usable but do not beat the stronger tree ensembles.
- These are useful baselines before moving to more sophisticated temporal or deep models.

## Feature Importance

Generated files:

- `outputs/feature_importance.csv`
- `outputs/logistic_coefficients.csv`

Top tree-based signals include:

- `relationships`
- `milestones`
- `age_last_milestone_year`
- `funding_total_usd`
- `age_first_milestone_year`
- `avg_participants`
- `age_last_funding_year`
- `age_first_funding_year`

The single most important decision-tree feature is:

- `relationships`

Interpretation:

- Network/relationship count is the clearest signal in the dataset.
- Funding and milestone maturity are also important.
- Industry and state matter, but the strongest model signal is from quantitative startup maturity metrics.

## Recommended Project Narrative

A defensible project framing is:

> We first establish that funding history, investor participation, milestone timing, geography, and industry features contain measurable signal for acquisition outcomes. Classical baselines achieve substantially better-than-majority performance, with Gradient Boosting reaching 0.8741 ROC-AUC on a stratified holdout set after removing the poorly defined `is_top500` feature. These results justify later work on richer temporal models, while also showing that simple structured startup features already capture meaningful acquisition relevance.

## Environment

Create the environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the full current workflow:

```bash
.venv/bin/python scripts/clean_and_test_dataset.py
.venv/bin/python scripts/exploratory_analysis_and_models.py
```

Dependencies:

- pandas
- numpy
- scipy
- scikit-learn

## Next Steps

Good next experiments:

- Hyperparameter tuning with nested cross-validation.
- Calibration curves for probability quality.
- SHAP or permutation importance for more interpretable feature effects.
- Time-aware splitting using founding/funding dates.
- Survival or hazard modeling for acquisition timing.
- Eventually, sequence or temporal models using funding and milestone timelines.
