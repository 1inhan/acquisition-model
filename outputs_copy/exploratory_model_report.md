# Startup Acquisition Baseline Model Report

## Target Distribution

| status | rows |
| --- | --- |
| acquired | 597 |
| closed | 326 |

## Holdout Results

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

## Cross-Validation Results

| model | split | accuracy_mean | accuracy_std | balanced_accuracy_mean | balanced_accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | 5_fold_stratified_cv | 0.7845 | 0.0392 | 0.7423 | 0.0424 | 0.8017 | 0.0284 | 0.8861 | 0.034 | 0.8417 | 0.0293 | 0.8235 | 0.0434 | 0.8648 | 0.029 |
| adaboost | 5_fold_stratified_cv | 0.7725 | 0.0303 | 0.7065 | 0.03 | 0.7668 | 0.017 | 0.9314 | 0.035 | 0.841 | 0.0228 | 0.8129 | 0.0252 | 0.8628 | 0.0177 |
| random_forest | 5_fold_stratified_cv | 0.7432 | 0.0283 | 0.7264 | 0.0318 | 0.8132 | 0.0269 | 0.784 | 0.033 | 0.7978 | 0.0232 | 0.8102 | 0.0296 | 0.864 | 0.0151 |
| svm_rbf | 5_fold_stratified_cv | 0.7367 | 0.0137 | 0.7229 | 0.0248 | 0.8144 | 0.028 | 0.7705 | 0.0247 | 0.7911 | 0.0083 | 0.8081 | 0.0251 | 0.8559 | 0.0205 |
| bagging_trees | 5_fold_stratified_cv | 0.7671 | 0.0369 | 0.738 | 0.0442 | 0.8101 | 0.0344 | 0.8376 | 0.0279 | 0.8232 | 0.0266 | 0.8018 | 0.0409 | 0.8526 | 0.0273 |
| hist_gradient_boosting | 5_fold_stratified_cv | 0.766 | 0.028 | 0.7246 | 0.036 | 0.7925 | 0.0277 | 0.866 | 0.0136 | 0.8274 | 0.0187 | 0.7902 | 0.0508 | 0.8391 | 0.0347 |
| logistic_regression | 5_fold_stratified_cv | 0.7129 | 0.0168 | 0.7147 | 0.0256 | 0.8244 | 0.0293 | 0.7085 | 0.0169 | 0.7616 | 0.0113 | 0.7818 | 0.0293 | 0.8475 | 0.0234 |
| svm_linear | 5_fold_stratified_cv | 0.7075 | 0.0232 | 0.7015 | 0.0316 | 0.807 | 0.0313 | 0.7219 | 0.0235 | 0.7615 | 0.0176 | 0.7726 | 0.0276 | 0.8442 | 0.0224 |
| extra_trees | 5_fold_stratified_cv | 0.6944 | 0.0345 | 0.6949 | 0.0298 | 0.8079 | 0.0234 | 0.6934 | 0.0606 | 0.7446 | 0.038 | 0.766 | 0.0366 | 0.8292 | 0.0196 |
| knn_7 | 5_fold_stratified_cv | 0.7432 | 0.0167 | 0.6902 | 0.0148 | 0.7649 | 0.0113 | 0.8711 | 0.0311 | 0.8142 | 0.0144 | 0.7606 | 0.033 | 0.8192 | 0.0306 |
| decision_tree | 5_fold_stratified_cv | 0.7108 | 0.049 | 0.6874 | 0.0366 | 0.7807 | 0.0182 | 0.7674 | 0.0816 | 0.7723 | 0.048 | 0.7478 | 0.0389 | 0.8108 | 0.0263 |
| gaussian_naive_bayes | 5_fold_stratified_cv | 0.43 | 0.1041 | 0.5107 | 0.0082 | 0.7861 | 0.1203 | 0.2335 | 0.3668 | 0.2304 | 0.2738 | 0.4576 | 0.0578 | 0.6481 | 0.0318 |

## Top Tree-Based Feature Importances

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
| decision_tree | categorical__industry_type_public_relations | 0 |
| decision_tree | categorical__industry_type_social | 0 |
| decision_tree | categorical__industry_type_search | 0 |
| decision_tree | categorical__industry_type_security | 0 |
| decision_tree | categorical__has_roundD_0 | 0 |

## Top Logistic Regression Coefficients

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
| categorical__industry_type_manufacturing | -0.4511 | 0.4511 |
| categorical__state_WI | -0.4462 | 0.4462 |
| categorical__state_TX | -0.4416 | 0.4416 |
| categorical__industry_type_biotech | 0.4399 | 0.4399 |
| categorical__state_CA | 0.4139 | 0.4139 |

## Notes

- Target is `is_acquired`, derived from `status == acquired`.
- `labels` and `status` are excluded from model inputs to avoid leakage.
- Metrics include accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.