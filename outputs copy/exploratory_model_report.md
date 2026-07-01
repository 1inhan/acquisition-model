# Startup Acquisition Baseline Model Report

## Target Distribution

| status | rows |
| --- | --- |
| acquired | 597 |
| closed | 326 |

## Holdout Results

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

## Cross-Validation Results

| model | split | accuracy_mean | accuracy_std | balanced_accuracy_mean | balanced_accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | 5_fold_stratified_cv | 0.7899 | 0.046 | 0.7506 | 0.0459 | 0.8082 | 0.0277 | 0.8845 | 0.0465 | 0.8445 | 0.0358 | 0.8236 | 0.0451 | 0.866 | 0.0309 |
| adaboost | 5_fold_stratified_cv | 0.7725 | 0.0282 | 0.7023 | 0.0287 | 0.7625 | 0.0164 | 0.9414 | 0.0307 | 0.8425 | 0.0206 | 0.8146 | 0.0251 | 0.8668 | 0.0216 |
| svm_rbf | 5_fold_stratified_cv | 0.7487 | 0.0194 | 0.7341 | 0.0248 | 0.821 | 0.0261 | 0.7839 | 0.0346 | 0.8012 | 0.0166 | 0.8142 | 0.0258 | 0.8641 | 0.0197 |
| random_forest | 5_fold_stratified_cv | 0.7476 | 0.0263 | 0.7319 | 0.03 | 0.8178 | 0.0266 | 0.7856 | 0.0309 | 0.8009 | 0.0214 | 0.8104 | 0.0315 | 0.8655 | 0.0148 |
| bagging_trees | 5_fold_stratified_cv | 0.7639 | 0.0413 | 0.7375 | 0.0446 | 0.8115 | 0.0322 | 0.8275 | 0.0393 | 0.8192 | 0.0319 | 0.8041 | 0.0388 | 0.8539 | 0.0278 |
| hist_gradient_boosting | 5_fold_stratified_cv | 0.7617 | 0.0365 | 0.7226 | 0.0429 | 0.7929 | 0.0315 | 0.856 | 0.0292 | 0.823 | 0.0266 | 0.7925 | 0.0465 | 0.8422 | 0.0326 |
| logistic_regression | 5_fold_stratified_cv | 0.7227 | 0.0237 | 0.7187 | 0.0294 | 0.8211 | 0.029 | 0.7319 | 0.03 | 0.7733 | 0.0198 | 0.7855 | 0.032 | 0.8501 | 0.0288 |
| svm_linear | 5_fold_stratified_cv | 0.714 | 0.0248 | 0.7058 | 0.0292 | 0.8075 | 0.0269 | 0.7336 | 0.0303 | 0.7683 | 0.0214 | 0.7776 | 0.031 | 0.8453 | 0.0306 |
| extra_trees | 5_fold_stratified_cv | 0.7226 | 0.037 | 0.7062 | 0.0279 | 0.7997 | 0.0153 | 0.762 | 0.0665 | 0.7789 | 0.0388 | 0.7716 | 0.0307 | 0.8368 | 0.0193 |
| knn_7 | 5_fold_stratified_cv | 0.7443 | 0.014 | 0.6882 | 0.0165 | 0.7622 | 0.0118 | 0.8794 | 0.0186 | 0.8165 | 0.0103 | 0.7604 | 0.0324 | 0.8242 | 0.0273 |
| decision_tree | 5_fold_stratified_cv | 0.7151 | 0.0382 | 0.6914 | 0.0297 | 0.7838 | 0.0172 | 0.7723 | 0.0645 | 0.7769 | 0.0356 | 0.7562 | 0.0307 | 0.8153 | 0.0239 |
| gaussian_naive_bayes | 5_fold_stratified_cv | 0.43 | 0.1041 | 0.5107 | 0.0082 | 0.7861 | 0.1203 | 0.2335 | 0.3668 | 0.2304 | 0.2738 | 0.4576 | 0.0578 | 0.6481 | 0.0318 |

## Top Tree-Based Feature Importances

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
| decision_tree | categorical__industry_type_search | 0 |
| decision_tree | categorical__industry_type_software | 0 |
| decision_tree | categorical__industry_type_semiconductor | 0 |
| decision_tree | categorical__has_roundA_0 | 0 |
| decision_tree | categorical__is_top500_0 | 0 |

## Top Logistic Regression Coefficients

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
| categorical__has_roundD_0 | -0.4652 | 0.4652 |
| categorical__state_TX | -0.4504 | 0.4504 |
| categorical__state_IL | -0.4397 | 0.4397 |
| categorical__industry_type_automotive | -0.4368 | 0.4368 |
| categorical__is_top500_1 | 0.3982 | 0.3982 |

## Notes

- Target is `is_acquired`, derived from `status == acquired`.
- `labels` and `status` are excluded from model inputs to avoid leakage.
- Metrics include accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.