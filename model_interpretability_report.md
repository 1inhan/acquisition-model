# Model Interpretability Report

## What can be inspected directly

- Decision Tree: explicit if/else rules are exported in `decision_tree_rules.txt`.
- Logistic Regression: signed coefficients are exported in `logistic_coefficients.csv`.
- Tree ensembles: aggregate feature importances are exported in `feature_importance.csv`.
- KNN, RBF SVM, Naive Bayes, and boosting models are less directly readable from simple rules.

## Model performance context

| model | split | train_rows | test_rows | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | stratified_holdout | 692 | 231 | 0.8095 | 0.7756 | 0.8261 | 0.8926 | 0.8581 | 0.8741 | 0.9201 |
| adaboost | stratified_holdout | 692 | 231 | 0.7879 | 0.7259 | 0.7778 | 0.9396 | 0.8511 | 0.8548 | 0.9062 |
| bagging_trees | stratified_holdout | 692 | 231 | 0.7489 | 0.745 | 0.837 | 0.7584 | 0.7958 | 0.8429 | 0.8912 |
| random_forest | stratified_holdout | 692 | 231 | 0.7489 | 0.7478 | 0.8421 | 0.7517 | 0.7943 | 0.8292 | 0.8843 |
| hist_gradient_boosting | stratified_holdout | 692 | 231 | 0.7662 | 0.7283 | 0.795 | 0.8591 | 0.8258 | 0.8213 | 0.875 |
| svm_rbf | stratified_holdout | 692 | 231 | 0.7229 | 0.7277 | 0.8346 | 0.7114 | 0.7681 | 0.8173 | 0.8701 |
| logistic_regression | stratified_holdout | 692 | 231 | 0.6883 | 0.7008 | 0.8235 | 0.6577 | 0.7313 | 0.7822 | 0.8517 |
| knn_7 | stratified_holdout | 692 | 231 | 0.7489 | 0.6984 | 0.7692 | 0.8725 | 0.8176 | 0.7703 | 0.836 |
| decision_tree | stratified_holdout | 692 | 231 | 0.7056 | 0.6923 | 0.7914 | 0.7383 | 0.7639 | 0.7685 | 0.8293 |
| extra_trees | stratified_holdout | 692 | 231 | 0.6407 | 0.6584 | 0.7946 | 0.5973 | 0.682 | 0.7625 | 0.8348 |
| svm_linear | stratified_holdout | 692 | 231 | 0.684 | 0.6892 | 0.8065 | 0.6711 | 0.7326 | 0.7617 | 0.838 |
| majority_baseline | stratified_holdout |  | 231 | 0.645 | 0.5 | 0.645 | 1 | 0.7842 | 0.5 | 0.645 |
| gaussian_naive_bayes | stratified_holdout | 692 | 231 | 0.355 | 0.4973 | 0.5 | 0.0067 | 0.0132 | 0.3807 | 0.5981 |

## Top feature importances

| model | feature | importance |
| --- | --- | --- |
| decision_tree | numeric__relationships | 0.5527 |
| decision_tree | numeric__avg_participants | 0.1393 |
| decision_tree | numeric__milestones | 0.1346 |
| decision_tree | numeric__funding_total_usd | 0.05913 |
| decision_tree | numeric__age_first_funding_year | 0.04664 |
| decision_tree | numeric__age_last_funding_year | 0.04354 |
| decision_tree | numeric__age_first_milestone_year | 0.01726 |
| decision_tree | categorical__industry_type_software | 0.006894 |
| decision_tree | categorical__industry_type_medical | 0 |
| decision_tree | categorical__industry_type_photo_video | 0 |
| decision_tree | categorical__industry_type_other | 0 |
| decision_tree | categorical__industry_type_news | 0 |
| decision_tree | categorical__industry_type_network_hosting | 0 |
| decision_tree | categorical__industry_type_music | 0 |
| decision_tree | categorical__industry_type_mobile | 0 |
| decision_tree | categorical__industry_type_messaging | 0 |
| decision_tree | categorical__industry_type_hardware | 0 |
| decision_tree | categorical__industry_type_manufacturing | 0 |
| decision_tree | categorical__industry_type_health | 0 |
| decision_tree | categorical__industry_type_real_estate | 0 |
| decision_tree | categorical__industry_type_games_video | 0 |
| decision_tree | categorical__industry_type_finance | 0 |
| decision_tree | categorical__industry_type_fashion | 0 |
| decision_tree | categorical__industry_type_enterprise | 0 |
| decision_tree | categorical__industry_type_education | 0 |

## Top logistic coefficients

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| categorical__state_NC | -1.649 | 1.649 |
| categorical__industry_type_other | -1.253 | 1.253 |
| categorical__state_OR | 1.238 | 1.238 |
| categorical__state_MA | 1.107 | 1.107 |
| categorical__industry_type_public_relations | -0.9542 | 0.9542 |
| categorical__industry_type_semiconductor | 0.9239 | 0.9239 |
| categorical__state_CT | -0.8135 | 0.8135 |
| categorical__industry_type_hardware | -0.7504 | 0.7504 |
| categorical__industry_type_cleantech | -0.749 | 0.749 |
| categorical__industry_type_ecommerce | -0.7023 | 0.7023 |
| categorical__industry_type_analytics | 0.6801 | 0.6801 |
| categorical__industry_type_news | 0.6585 | 0.6585 |
| numeric__relationships | 0.651 | 0.651 |
| categorical__industry_type_music | 0.6146 | 0.6146 |
| categorical__industry_type_photo_video | 0.5852 | 0.5852 |
| categorical__state_MD | 0.5798 | 0.5798 |
| categorical__state_OH | -0.5745 | 0.5745 |
| categorical__state_NY | 0.5722 | 0.5722 |
| categorical__state_CO | 0.5666 | 0.5666 |
| categorical__industry_type_search | -0.5229 | 0.5229 |
| categorical__has_roundD_1 | 0.5146 | 0.5146 |
| numeric__milestones | 0.5129 | 0.5129 |
| categorical__industry_type_health | 0.5017 | 0.5017 |
| categorical__has_roundD_0 | -0.4689 | 0.4689 |
| categorical__industry_type_travel | 0.4522 | 0.4522 |

## Decision tree leaf summary

| leaf_id | rows | actual_acquired_rate | predicted_acquired_rate | avg_predicted_probability |
| --- | --- | --- | --- | --- |
| 27 | 71 | 0.831 | 1 | 0.7354 |
| 5 | 20 | 0.1 | 0 | 0 |
| 28 | 18 | 0.9444 | 1 | 0.9196 |
| 10 | 14 | 0.5714 | 1 | 0.5545 |
| 17 | 14 | 0.7857 | 0 | 0.1323 |
| 18 | 13 | 0.6154 | 1 | 0.5214 |
| 25 | 11 | 0.8182 | 1 | 0.6854 |
| 12 | 11 | 0.2727 | 0 | 0.3359 |
| 24 | 10 | 0.8 | 0 | 0.4145 |
| 21 | 10 | 0.8 | 0 | 0.4119 |
| 9 | 10 | 0.5 | 0 | 0.2463 |
| 7 | 9 | 0.2222 | 0 | 0.1789 |
| 6 | 8 | 0 | 0 | 0.1454 |
| 19 | 6 | 0.8333 | 1 | 0.7212 |
| 13 | 6 | 0.6667 | 1 | 0.6354 |

## RNN note

No RNN has been trained yet. The current dataset is row-level tabular data, not a true sequence dataset. An RNN would make more sense after converting each startup into a timestamped funding/milestone event sequence.