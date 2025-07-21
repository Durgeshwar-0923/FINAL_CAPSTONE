.
├── Dockerfile
├── README.md
├── Untitled.ipynb
├── airflow
│   ├── dags
│   │   ├── data_pipeline.py
│   │   └── retrain_pipeline.py
│   └── plugins
├── catboost_info
│   ├── catboost_training.json
│   ├── learn
│   │   └── events.out.tfevents
│   ├── learn_error.tsv
│   └── time_left.tsv
├── data
│   ├── drift_reports
│   ├── processed
│   │   └── 13_final_features.csv
│   └── raw
│       ├── Lead Scoring.csv
│       └── staged_lead_scoring_data.csv
├── deploy.py
├── docker
│   ├── app.py
│   ├── docker-compose.yml
│   ├── nginx.conf
│   ├── preprocessor.py
│   ├── serve
│   ├── untitled.py
│   └── wsgi.py
├── evidently_reports
│   ├── evidently_prediction_vs_training_cleaned_2025-07-18_14-48-05.json
│   ├── evidently_prediction_vs_training_cleaned_2025-07-18_14-49-53.html
│   ├── evidently_prediction_vs_training_cleaned_2025-07-18_14-49-53.json
│   ├── evidently_train_vs_test_2025-07-18_14-10-12.html
│   ├── evidently_train_vs_test_2025-07-18_14-10-12.json
│   ├── evidently_train_vs_test_2025-07-19_14-18-35.html
│   ├── evidently_train_vs_test_2025-07-19_14-18-35.json
│   ├── evidently_train_vs_test_2025-07-19_14-30-47.html
│   ├── evidently_train_vs_test_2025-07-19_14-30-47.json
│   ├── evidently_train_vs_test_2025-07-19_14-34-11.html
│   ├── evidently_train_vs_test_2025-07-19_14-34-11.json
│   ├── evidently_train_vs_test_2025-07-19_14-41-16.html
│   ├── evidently_train_vs_test_2025-07-19_14-41-16.json
│   ├── evidently_train_vs_test_2025-07-19_14-48-51.html
│   ├── evidently_train_vs_test_2025-07-19_14-48-51.json
│   ├── evidently_train_vs_test_2025-07-19_15-02-14.html
│   ├── evidently_train_vs_test_2025-07-19_15-02-14.json
│   ├── evidently_train_vs_test_2025-07-19_15-05-27.html
│   ├── evidently_train_vs_test_2025-07-19_15-05-27.json
│   ├── evidently_train_vs_test_2025-07-19_15-19-06.html
│   ├── evidently_train_vs_test_2025-07-19_15-19-06.json
│   ├── evidently_train_vs_test_2025-07-19_15-43-20.html
│   ├── evidently_train_vs_test_2025-07-19_15-43-20.json
│   ├── evidently_train_vs_test_2025-07-19_15-49-54.html
│   ├── evidently_train_vs_test_2025-07-19_15-49-54.json
│   ├── evidently_train_vs_test_2025-07-19_16-08-50.html
│   ├── evidently_train_vs_test_2025-07-19_16-08-50.json
│   ├── evidently_train_vs_test_2025-07-19_16-18-14.html
│   ├── evidently_train_vs_test_2025-07-19_16-18-14.json
│   ├── evidently_train_vs_test_2025-07-19_16-22-01.html
│   ├── evidently_train_vs_test_2025-07-19_16-24-15.html
│   └── evidently_train_vs_test_2025-07-19_16-24-15.json
├── flask.txt
├── notebooks
│   ├── exploratory_analysis.ipynb
│   └── exploratory_analysis.py
├── outputs
│   ├── catboostclassifier_feature_importances.png
│   ├── catboostclassifier_optuna_trials.csv
│   ├── gradientboostingclassifier_feature_importances.png
│   ├── gradientboostingclassifier_optuna_trials.csv
│   ├── lgbmclassifier_feature_importances.png
│   ├── lgbmclassifier_optuna_trials.csv
│   ├── logisticregression_optuna_trials.csv
│   ├── model_summary.csv
│   ├── optuna_catboostclassifier_history.html
│   ├── optuna_gradientboostingclassifier_history.html
│   ├── optuna_lgbmclassifier_history.html
│   ├── optuna_logisticregression_history.html
│   ├── optuna_randomforestclassifier_history.html
│   ├── optuna_xgbclassifier_history.html
│   ├── randomforestclassifier_feature_importances.png
│   ├── randomforestclassifier_optuna_trials.csv
│   ├── shap_summary_plot.png
│   ├── stages
│   │   ├── 10_scale.csv
│   │   ├── 11_compute_shap.csv
│   │   ├── 12_cluster.csv
│   │   ├── 13_shap_feature_selection.csv
│   │   ├── 13_shap_feature_selection_removed_columns.json
│   │   ├── 1_clean.csv
│   │   ├── 1_clean_removed_columns.json
│   │   ├── 2_type_convert.csv
│   │   ├── 3_impute.csv
│   │   ├── 4_outlier.csv
│   │   ├── 5_engineer.csv
│   │   ├── 5_engineer_removed_columns.json
│   │   ├── 6_bin.csv
│   │   ├── 7_rare_label.csv
│   │   ├── 8_vif.csv
│   │   ├── 8_vif_removed_columns.json
│   │   ├── 9_encode.csv
│   │   └── 9_encode_removed_columns.json
│   ├── xgbclassifier_feature_importances.png
│   └── xgbclassifier_optuna_trials.csv
├── reports
│   └── eda_report.html
├── requirements.txt
├── requirementstwo.txt
├── run.py
├── sagemaker
│   ├── 0
│   │   └── meta.yaml
│   └── 644704234141752827
│       ├── d50dce588b104d38a181d7a8060543a6
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   │   └── test_metric
│       │   ├── params
│       │   │   └── test_param
│       │   └── tags
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.git.commit
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       └── mlflow.user
│       └── meta.yaml
├── setup.py
├── src
│   ├── api
│   │   ├── app.py
│   │   ├── prediction_service.py
│   │   └── templates
│   │       ├── index.html
│   │       └── result.html
│   ├── config
│   │   └── config.py
│   ├── data_ingestion
│   │   ├── data_loader.py
│   │   └── database_operations.py
│   ├── data_processing
│   │   ├── eda.py
│   │   ├── feature_selector.py
│   │   └── preprocessor
│   │       ├── binning.py
│   │       ├── cleaning.py
│   │       ├── clustering.py
│   │       ├── encoding.py
│   │       ├── feature_engineering.py
│   │       ├── feature_selection.py
│   │       ├── missing_imputation.py
│   │       ├── outlier_handler.py
│   │       ├── pipeline.py
│   │       ├── rare_label_encoder.py
│   │       ├── scaling.py
│   │       ├── target_encoder_wrapper.py
│   │       ├── type_conversion.py
│   │       └── vif_filter.py
│   ├── models
│   │   ├── optuna_tuner.py
│   │   └── train_models.py
│   ├── monitoring
│   │   ├── drift_detector.py
│   │   └── model_monitor.py
│   └── utils
│       ├── logger.py
│       ├── metrics.py
│       └── mlflow_logger.py
├── structure.txt
├── test.py
├── tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data_processing.py
│   └── test_models.py
├── untitled1.txt
└── your_mlflow_tracking_uri_here
  
