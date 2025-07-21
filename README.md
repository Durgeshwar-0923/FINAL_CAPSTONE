workspace strucuture
```text.
.
├── Dockerfile                 # Instructions to build the container image for deployment
├── README.md                  # Project overview and instructions
├── Untitled.ipynb             # Scratch or testing notebook
├── airflow/                   # Airflow orchestration for automation
│   ├── dags/                  # DAGs: data pipeline & retraining workflows
│   │   ├── data_pipeline.py   # DAG for ETL, loading new data
│   │   └── retrain_pipeline.py # DAG for triggering model retraining on drift
│   └── plugins/               # Custom Airflow plugins (empty placeholder or custom ops)
├── catboost_info/             # CatBoost training logs and metadata
│   ├── catboost_training.json # CatBoost training info
│   ├── learn/                 # CatBoost learn events
│   ├── learn_error.tsv        # CatBoost training error metrics
│   └── time_left.tsv          # Estimated time left for training
├── data/                      # Data storage
│   ├── drift_reports/         # Drift report outputs (e.g., Evidently)
│   ├── processed/             # Processed datasets ready for modeling
│   │   └── 13_final_features.csv # Final features dataset
│   └── raw/                   # Raw input data files
│       ├── Lead Scoring.csv   # Original lead scoring data
│       └── staged_lead_scoring_data.csv # Data staged for ETL
├── deploy.py                  # Script for deploying model container to ECR/SageMaker
├── docker/                    # Docker deployment files
│   ├── app.py                 # Flask app for serving predictions
│   ├── docker-compose.yml     # Docker Compose config for local multi-container runs
│   ├── nginx.conf             # NGINX config for reverse proxying Flask app
│   ├── preprocessor.py        # Preprocessing logic for incoming requests
│   ├── serve/                 # (Likely) Static/WSGI files for serving app
│   ├── untitled.py            # (Likely) Placeholder or scratch file
│   └── wsgi.py                # WSGI entry point for Flask in production
├── evidently_reports/         # Generated drift and data quality reports by Evidently
│   ├── *.json, *.html         # Different timestamps: train vs test, prediction vs training
├── flask.txt                  # Requirements for Flask app (or notes)
├── notebooks/                 # Jupyter notebooks for EDA and experimentation
│   ├── exploratory_analysis.ipynb # Initial EDA notebook
│   └── exploratory_analysis.py    # Script version of EDA
├── outputs/                   # Outputs: plots, feature importances, Optuna trials, shap plots
│   ├── *.png                  # Feature importance, SHAP summary plots
│   ├── *.csv                  # Optuna tuning results, model summaries
│   ├── *.html                 # Optuna history visualizations
│   ├── stages/                # Intermediate preprocessing outputs for pipeline stages
│   │   ├── *.csv              # Staged feature engineering steps
│   │   ├── *.json             # Columns removed/selected at each stage
├── reports/                   # EDA or final project reports
│   └── eda_report.html        # Full EDA report in HTML
├── requirements.txt           # Project dependencies
├── requirementstwo.txt        # Alternate requirements file (or dev env)
├── run.py                     # Entry script to run the project locally
├── sagemaker/                 # SageMaker experiment tracking outputs (MLflow)
│   ├── */                     # Run folders with meta.yaml and metrics
│   └── meta.yaml              # Experiment metadata
├── setup.py                   # Python package setup if needed for pip install
├── src/                       # Core source code
│   ├── api/                   # Flask API logic
│   │   ├── app.py             # Main API entry point
│   │   ├── prediction_service.py # Handles prediction calls
│   │   └── templates/         # HTML templates for UI
│   │       ├── index.html     # Input form page
│   │       └── result.html    # Result display page
│   ├── config/                # Configuration files
│   │   └── config.py          # App and pipeline configs
│   ├── data_ingestion/        # Data ingestion layer
│   │   ├── data_loader.py     # Load data from S3/Redshift
│   │   └── database_operations.py # DB ops for Redshift
│   ├── data_processing/       # Data cleaning and feature engineering
│   │   ├── eda.py             # EDA utilities
│   │   ├── feature_selector.py # Feature selection logic
│   │   └── preprocessor/      # Detailed preprocessing modules
│   │       ├── binning.py, cleaning.py, clustering.py, etc.
│   ├── models/                # Model training & tuning
│   │   ├── optuna_tuner.py    # Hyperparameter tuning with Optuna
│   │   └── train_models.py    # Training logic
│   ├── monitoring/            # Model & data drift monitoring
│   │   ├── drift_detector.py  # Drift detection logic
│   │   └── model_monitor.py   # General monitoring tasks
│   └── utils/                 # Helper utilities
│       ├── logger.py          # Logging utility
│       ├── metrics.py         # Custom metrics calculation
│       └── mlflow_logger.py   # MLflow logging utilities
├── structure.txt              # Text description of project structure (this!)
├── test.py                    # Example or scratch test script
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_api.py            # API tests
│   ├── test_data_processing.py # Preprocessing tests
│   └── test_models.py         # Model unit tests
├── untitled1.txt              # Notes or scratch
└── your_mlflow_tracking_uri_here # Placeholder for MLflow tracking URI

  ```
This repository contains a scalable, production-ready machine learning pipeline designed to automate lead scoring and help sales teams prioritize high-value leads more effectively. Built entirely on AWS cloud services, the system ensures robust ETL, reliable training, automated deployment, and continuous monitoring with MLOps best practices.

---

## 🧐 Business Problem
Sales teams often spend significant time and resources following up with low-quality leads, leading to lost opportunities and lower conversion rates. This project solves that by building an accurate lead scoring model that continuously adapts to data changes, helping the business focus efforts where they matter most.

---

## 🎯 Objectives
- Automate lead scoring with a reliable ML model.
- Maintain high prediction accuracy (AUC > 0.97).
- Ensure secure, scalable data storage and processing.
- Automate deployment, monitoring, and retraining to handle drift.
- Deliver predictions through an easy-to-use web interface.

---

## 🗂️ Architecture Overview

**Key stages of the pipeline:**

1. **Data Storage:**  
   - Raw and new prediction data is stored in **Amazon S3**.

2. **ETL & Data Cataloging:**  
   - **AWS Glue Crawlers** scan the S3 bucket to create a Data Catalog.
   - **AWS Glue ETL Jobs** transform and load data into **Amazon Redshift**.

3. **Model Development:**  
   - **Amazon SageMaker** connects to Redshift, loads data, performs preprocessing, and trains multiple models.
   - **MLflow Tracking Server** logs experiments, metrics, and artifacts.

4. **Model Versioning:**  
   - The best-performing model is registered in the **MLflow Model Registry** with a `Champion` alias.

5. **Containerization & Deployment:**  
   - The model and artifacts are packaged using **Docker** and **NGINX**, pushed to **Amazon ECR**.
   - **SageMaker Endpoints** deploy the latest container for real-time inference.

6. **Serving:**  
   - A **Flask UI** connects to the SageMaker Endpoint, allowing users to submit leads and get predictions.

7. **Monitoring & Retraining:**  
   - **Apache Airflow** schedules drift detection and automatically triggers retraining and redeployment when needed.

---

## ⚙️ Tech Stack

- **AWS S3** — Raw data storage  
- **AWS Glue** — ETL and data cataloging  
- **Amazon Redshift** — Data warehouse  
- **Amazon SageMaker** — Model development & hosting  
- **MLflow** — Experiment tracking & model registry  
- **Docker & NGINX** — Containerization  
- **Amazon ECR** — Container registry  
- **Apache Airflow** — Orchestration & monitoring  
- **Flask** — User-facing prediction UI  
- **Python** — Core language for data processing & modeling
