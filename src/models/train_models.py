import os
import tarfile
import boto3
import mlflow
import shutil
import sagemaker
import warnings
import atexit
import tempfile
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table

# --- Custom Module Imports (ensure these paths are correct) ---
from src.utils.logger import setup_logger
from src.utils.metrics import classification_report_dict
from src.models.optuna_tuner import optimize_model
from src.data_processing.preprocessor.target_encoder_wrapper import TargetEncoderWrapper
from src.monitoring.drift_detector import log_drift_report

# --- Initial Setup & Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg")

logger = setup_logger(__name__)
console = Console()

# --- Configuration ---
S3_BUCKET_NAME = "flaskcapstonebucket"
MODEL_NAME_IN_REGISTRY = "LeadConversionModel"
PROJECT_ROOT = os.path.expanduser("~/FINAL_CAPSTONE")
LOCAL_ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")
STAGING_DIR = "staging"
MODEL_ARTIFACT_BUNDLE = "model.tar.gz"

MLFLOW_TRACKING_ARN = "arn:aws:sagemaker:ap-south-1:439932142398:mlflow-tracking-server/capstonetracking"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_ARN
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.ap-south-1.amazonaws.com"

# ==============================================================================
# SECTION 1: MODEL TRAINING LOGIC (from train_models.py)
# ==============================================================================

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression, "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier, "LightGBM": LGBMClassifier, "CatBoost": CatBoostClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

TUNING_SPACE = {
    "LogisticRegression": lambda t: {"C": t.suggest_float("C", 0.01, 10.0)},
    "RandomForest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200), "max_depth": t.suggest_int("max_depth", 3, 15)},
    "XGBoost": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3), "max_depth": t.suggest_int("max_depth", 3, 10)},
    "LightGBM": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2), "num_leaves": t.suggest_int("num_leaves", 20, 150)},
    "CatBoost": lambda t: {"iterations": t.suggest_int("iterations", 50, 200), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3), "depth": t.suggest_int("depth", 3, 10)},
    "GradientBoosting": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3), "max_depth": t.suggest_int("max_depth", 3, 10)},
}

def safe_start_run(name=None, nested=False):
    if mlflow.active_run():
        mlflow.end_run()
    return mlflow.start_run(run_name=name, nested=nested)

def train_all_models(
    data_path: str = "data/processed/13_final_features.csv",
    experiment_name: str = "Lead_Conversion_Modeling",
    target: str = "Converted",
    test_size: float = 0.2,
    n_trials: int = 15
):
    mlflow.set_experiment(experiment_name)
    df = pd.read_csv(data_path)
    logger.info(f"Loaded processed data: {df.shape}")

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    try:
        logger.info("Filtering columns for drift detection...")
        X_train_numeric = X_train.select_dtypes(include=np.number)
        X_test_numeric = X_test.select_dtypes(include=np.number)
        train_variant_cols = X_train_numeric.columns[X_train_numeric.std() > 0]
        test_variant_cols = X_test_numeric.columns[X_test_numeric.std() > 0]
        safe_numeric_cols = train_variant_cols.intersection(test_variant_cols)
        X_train_drift = pd.concat([X_train[safe_numeric_cols], X_train.select_dtypes(exclude=np.number)], axis=1)
        X_test_drift = pd.concat([X_test[safe_numeric_cols], X_test.select_dtypes(exclude=np.number)], axis=1)
        log_drift_report(X_train_drift, X_test_drift, dataset_name="train_vs_test")
    except Exception as e:
        logger.warning(f"⚠️ Could not perform drift detection. Error: {e}")

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = TargetEncoderWrapper(cols_to_encode=cat_cols)
    X_train_enc = encoder.fit_transform(X_train.copy(), y_train)
    X_test_enc = encoder.transform(X_test.copy())
    joblib.dump(encoder, os.path.join(LOCAL_ARTIFACTS_PATH, "final_target_encoder.joblib"))
    logger.info(f"💾 Saved final target encoder to artifacts/final_target_encoder.joblib")

    scaler = StandardScaler().fit(X_train_enc)
    X_train_enc_df = pd.DataFrame(scaler.transform(X_train_enc), columns=X_train_enc.columns)
    X_test_enc_df = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns)
    joblib.dump(scaler, os.path.join(LOCAL_ARTIFACTS_PATH, "feature_scaler.pkl"))
    logger.info(f"💾 Saved final feature scaler to artifacts/feature_scaler.pkl")

    results, models = [], []
    best_auc, best_model_name, best_run_id, best_model_object = 0.0, None, None, None

    with safe_start_run("All_Model_Training") as main_run:
        for name, Cls in CLASSIFIERS.items():
            with mlflow.start_run(run_name=name, nested=True) as nested_run:
                params = optimize_model(Cls, TUNING_SPACE[name], X_train_enc_df, y_train, n_trials=n_trials)
                if name == "CatBoost": params['verbose'] = 0
                model = Cls(**params)
                model.fit(X_train_enc_df, y_train)
                probs = model.predict_proba(X_test_enc_df)[:, 1]
                metrics = {"roc_auc": roc_auc_score(y_test, probs)}
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                input_example = X_train_enc_df.head(5)
                mlflow.sklearn.log_model(model, name, input_example=input_example)

                if metrics["roc_auc"] > best_auc:
                    best_auc, best_model_name, best_run_id, best_model_object = metrics["roc_auc"], name, nested_run.info.run_id, model
                results.append({"model": name, **metrics})
                models.append((name, model))

        with mlflow.start_run(run_name="StackingEnsemble", nested=True) as stack_run:
            stack = StackingClassifier(estimators=models, final_estimator=LogisticRegression(), cv=StratifiedKFold(5), n_jobs=1, passthrough=True)
            stack.fit(X_train_enc_df, y_train)
            probs = stack.predict_proba(X_test_enc_df)[:, 1]
            stack_metrics = {"roc_auc": roc_auc_score(y_test, probs)}
            mlflow.log_metrics(stack_metrics)
            input_example = X_train_enc_df.head(5)
            mlflow.sklearn.log_model(stack, "StackingEnsemble", input_example=input_example)

            if stack_metrics["roc_auc"] > best_auc:
                best_model_name, best_run_id, best_model_object = "StackingEnsemble", stack_run.info.run_id, stack
            results.append({"model": "StackingEnsemble", **stack_metrics})

        if best_model_object:
            logger.info(f"Calculating SHAP values for the best model: {best_model_name}")
            try:
                # ... (SHAP logic remains the same) ...
                mlflow.log_artifact("outputs/shap_summary_plot.png", "explainability")
            except Exception as e:
                logger.warning(f"⚠️ Could not generate SHAP plot. Error: {e}")

    if best_model_name and best_run_id:
        client = MlflowClient()
        run_model_uri = f"runs:/{best_run_id}/{best_model_name}"
        mv = mlflow.register_model(run_model_uri, MODEL_NAME_IN_REGISTRY)
        client.set_registered_model_alias(name=MODEL_NAME_IN_REGISTRY, alias="champion", version=mv.version)
        logger.info("✅ Model registration and promotion complete.")

    console.print(pd.DataFrame(results))
    return best_model_name

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # --- Part 1: Train the model ---
    print("\n--- Part 1: Starting Model Training ---")
    # This assumes your run.py has already created the processed data file
    processed_data_file = os.path.join(PROJECT_ROOT, "data/processed/13_final_features.csv")
    if not os.path.exists(processed_data_file):
        raise FileNotFoundError(f"Processed data not found at {processed_data_file}. Please run the preprocessing pipeline first.")
    
    train_all_models(data_path=processed_data_file)
    print("\n🎉 Training complete!")

    # --- Part 2: Package the model ---
    print("\n--- Part 2: Starting Model Packaging ---")
    
    # --- Step 1: Download the "Production" model from the MLflow Model Registry ---
    print("\n--- Step 1: Downloading Production Model ---")
    client = MlflowClient()
    try:
        production_model = client.get_model_version_by_alias(name=MODEL_NAME_IN_REGISTRY, alias="champion")
        model_uri = production_model.source
        print(f"Found production model: Version {production_model.version}, Run ID {production_model.run_id}")

        if os.path.exists(STAGING_DIR): shutil.rmtree(STAGING_DIR)
        os.makedirs(STAGING_DIR)

        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=STAGING_DIR)
        print(f"✅ Successfully downloaded model to '{STAGING_DIR}/'")
    except Exception as e:
        print(f"❌ Failed to download model. Error: {e}")
        raise e

    # --- Step 2: Copy all preprocessing artifacts into the package ---
    print("\n--- Step 2: Copying Preprocessing Artifacts ---")
    try:
        destination_path = os.path.join(STAGING_DIR, "artifacts")
        shutil.copytree(LOCAL_ARTIFACTS_PATH, destination_path)
        print(f"✅ Successfully copied all preprocessing artifacts to '{destination_path}/'")
    except Exception as e:
        print(f"❌ Failed to copy artifacts. Error: {e}")
        raise e

    # --- Step 3: Create the compressed model.tar.gz file ---
    print("\n--- Step 3: Creating model.tar.gz ---")
    try:
        with tarfile.open(MODEL_ARTIFACT_BUNDLE, "w:gz") as tar:
            tar.add(STAGING_DIR, arcname='.')
        print(f"✅ Successfully created '{MODEL_ARTIFACT_BUNDLE}'")
    except Exception as e:
        print(f"❌ Failed to create tarball. Error: {e}")
        raise e

    # --- Step 4: Upload the model package to S3 ---
    print("\n--- Step 4: Uploading to S3 ---")
    try:
        s3_client = boto3.client('s3')
        s3_model_path = f"models/{MODEL_NAME_IN_REGISTRY}/model.tar.gz"
        s3_client.upload_file(MODEL_ARTIFACT_BUNDLE, S3_BUCKET_NAME, s3_model_path)
        model_s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_model_path}"
        print(f"✅ Successfully uploaded model package to: {model_s3_uri}")
    except Exception as e:
        print(f"❌ Failed to upload to S3. Error: {e}")
        raise e

    print("\n🎉 Full Train-and-Package pipeline complete!")
