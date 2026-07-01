# Model Interpretability Report

## What can be inspected directly

- Decision Tree: explicit if/else rules are exported in `decision_tree_rules.txt`.
- Logistic Regression: signed coefficients are exported in `logistic_coefficients.csv`.
- Tree ensembles: aggregate feature importances are exported in `feature_importance.csv`.
- KNN, RBF SVM, Naive Bayes, and boosting models are less directly readable from simple rules.

## Model performance context

| model | split | train_rows | test_rows | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | stratified_holdout | 692 | 231 | 0.8095 | 0.7783 | 0.8302 | 0.8859 | 0.8571 | 0.8694 | 0.9159 |
| adaboost | stratified_holdout | 692 | 231 | 0.7879 | 0.7232 | 0.7747 | 0.9463 | 0.852 | 0.8611 | 0.9118 |
| bagging_trees | stratified_holdout | 692 | 231 | 0.7576 | 0.7572 | 0.8496 | 0.7584 | 0.8014 | 0.8465 | 0.8953 |
| random_forest | stratified_holdout | 692 | 231 | 0.7706 | 0.7646 | 0.8478 | 0.7852 | 0.8153 | 0.8353 | 0.8899 |
| svm_rbf | stratified_holdout | 692 | 231 | 0.7489 | 0.745 | 0.837 | 0.7584 | 0.7958 | 0.8218 | 0.872 |
| hist_gradient_boosting | stratified_holdout | 692 | 231 | 0.7792 | 0.7411 | 0.8025 | 0.8725 | 0.836 | 0.8174 | 0.8721 |
| logistic_regression | stratified_holdout | 692 | 231 | 0.697 | 0.6993 | 0.811 | 0.6913 | 0.7464 | 0.7954 | 0.8586 |
| decision_tree | stratified_holdout | 692 | 231 | 0.7056 | 0.6923 | 0.7914 | 0.7383 | 0.7639 | 0.7735 | 0.8309 |
| svm_linear | stratified_holdout | 692 | 231 | 0.684 | 0.681 | 0.7923 | 0.6913 | 0.7384 | 0.7713 | 0.8426 |
| extra_trees | stratified_holdout | 692 | 231 | 0.7143 | 0.7072 | 0.8074 | 0.7315 | 0.7676 | 0.7644 | 0.833 |
| knn_7 | stratified_holdout | 692 | 231 | 0.7446 | 0.6978 | 0.7711 | 0.8591 | 0.8127 | 0.7546 | 0.8166 |
| majority_baseline | stratified_holdout |  | 231 | 0.645 | 0.5 | 0.645 | 1 | 0.7842 | 0.5 | 0.645 |
| gaussian_naive_bayes | stratified_holdout | 692 | 231 | 0.355 | 0.4973 | 0.5 | 0.0067 | 0.0132 | 0.3807 | 0.5981 |

## Top feature importances

| model | feature | importance |
| --- | --- | --- |
| decision_tree | numeric__relationships | 0.5579 |
| decision_tree | numeric__avg_participants | 0.1398 |
| decision_tree | numeric__milestones | 0.1351 |
| decision_tree | numeric__funding_total_usd | 0.05936 |
| decision_tree | numeric__age_first_funding_year | 0.04682 |
| decision_tree | numeric__age_last_funding_year | 0.04373 |
| decision_tree | numeric__age_first_milestone_year | 0.01732 |
| decision_tree | categorical__industry_type_messaging | 0 |
| decision_tree | categorical__industry_type_real_estate | 0 |
| decision_tree | categorical__industry_type_public_relations | 0 |
| decision_tree | categorical__industry_type_photo_video | 0 |
| decision_tree | categorical__industry_type_other | 0 |
| decision_tree | categorical__industry_type_news | 0 |
| decision_tree | categorical__industry_type_network_hosting | 0 |
| decision_tree | categorical__industry_type_music | 0 |
| decision_tree | categorical__industry_type_mobile | 0 |
| decision_tree | categorical__industry_type_health | 0 |
| decision_tree | categorical__industry_type_medical | 0 |
| decision_tree | categorical__industry_type_manufacturing | 0 |
| decision_tree | categorical__industry_type_security | 0 |
| decision_tree | categorical__industry_type_hardware | 0 |
| decision_tree | categorical__industry_type_games_video | 0 |
| decision_tree | categorical__industry_type_finance | 0 |
| decision_tree | categorical__industry_type_fashion | 0 |
| decision_tree | categorical__industry_type_enterprise | 0 |

## Top logistic coefficients

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| categorical__state_NC | -1.51 | 1.51 |
| categorical__state_OR | 1.391 | 1.391 |
| categorical__industry_type_other | -1.292 | 1.292 |
| categorical__state_MA | 0.9561 | 0.9561 |
| categorical__industry_type_public_relations | -0.9287 | 0.9287 |
| categorical__industry_type_semiconductor | 0.8972 | 0.8972 |
| categorical__industry_type_hardware | -0.7874 | 0.7874 |
| categorical__state_CT | -0.7574 | 0.7574 |
| categorical__industry_type_music | 0.7154 | 0.7154 |
| categorical__industry_type_cleantech | -0.7115 | 0.7115 |
| categorical__industry_type_analytics | 0.6987 | 0.6987 |
| categorical__industry_type_ecommerce | -0.6577 | 0.6577 |
| numeric__relationships | 0.6377 | 0.6377 |
| categorical__industry_type_news | 0.6064 | 0.6064 |
| categorical__state_OH | -0.6003 | 0.6003 |
| categorical__industry_type_health | 0.5812 | 0.5812 |
| categorical__state_MD | 0.577 | 0.577 |
| categorical__industry_type_search | -0.565 | 0.565 |
| categorical__industry_type_photo_video | 0.5645 | 0.5645 |
| numeric__milestones | 0.5053 | 0.5053 |
| categorical__state_CO | 0.5017 | 0.5017 |
| categorical__state_NY | 0.5014 | 0.5014 |
| categorical__has_roundD_1 | 0.4766 | 0.4766 |
| categorical__industry_type_manufacturing | -0.4684 | 0.4684 |
| categorical__industry_type_biotech | 0.4669 | 0.4669 |

## Decision tree leaf summary

| leaf_id | rows | actual_acquired_rate | predicted_acquired_rate | avg_predicted_probability |
| --- | --- | --- | --- | --- |
| 27 | 71 | 0.831 | 1 | 0.7354 |
| 28 | 18 | 0.9444 | 1 | 0.9196 |
| 5 | 16 | 0 | 0 | 0.0102 |
| 10 | 14 | 0.5714 | 1 | 0.5545 |
| 17 | 14 | 0.7857 | 0 | 0.1323 |
| 18 | 13 | 0.6154 | 1 | 0.5214 |
| 25 | 11 | 0.8182 | 1 | 0.6854 |
| 12 | 11 | 0.2727 | 0 | 0.3359 |
| 7 | 11 | 0.0909 | 0 | 0.1789 |
| 24 | 10 | 0.8 | 0 | 0.4145 |
| 21 | 10 | 0.8 | 0 | 0.4119 |
| 9 | 10 | 0.5 | 0 | 0.2463 |
| 6 | 10 | 0.3 | 0 | 0.1029 |
| 19 | 6 | 0.8333 | 1 | 0.7212 |
| 13 | 6 | 0.6667 | 1 | 0.6354 |

## RNN note

No RNN has been trained yet. The current dataset is row-level tabular data, not a true sequence dataset. An RNN would make more sense after converting each startup into a timestamped funding/milestone event sequence.