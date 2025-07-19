# src/models/train_models.py

import warnings, os, atexit, shutil, tempfile, contextlib
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Tcl_AsyncDelete: async handler deleted by the wrong thread")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=UserWarning, message=".*distutils.*")

os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""

try:
    import tkinter as _tk
    if hasattr(_tk, "Image"):
        _tk.Image.__del__ = lambda self: None
    if hasattr(_tk, "Variable"):
        _tk.Variable.__del__ = lambda self: None
    if hasattr(_tk, "Misc"):
        _tk.Misc.__del__ = lambda self: None
except Exception:
    pass

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

_temp_dir = tempfile.mkdtemp()
os.environ["JOBLIB_TEMP_FOLDER"] = _temp_dir

def cleanup_temp_dir():
    try:
        shutil.rmtree(_temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning during temp cleanup: {e}")
atexit.register(cleanup_temp_dir)

# ─── Core Libraries ───────────────────────────────────────────────────
import mlflow
import numpy as np
import pandas as pd
import joblib
import shap # --- ADDITION: Import SHAP ---
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

# ─── Progress & Visualization ─────────────────────────────────────────
from rich.console import Console
from rich.table import Table

# ─── Custom Modules ──────────────────────────────────────────────────
from src.utils.logger import setup_logger
from src.utils.metrics import classification_report_dict
from src.models.optuna_tuner import optimize_model
from src.data_processing.preprocessor.target_encoder_wrapper import TargetEncoderWrapper
from src.monitoring.drift_detector import log_drift_report

# ─── Logger & Setup ──────────────────────────────────────────────────
logger = setup_logger(__name__)
console = Console()
client = MlflowClient()

for d in ["artifacts", "drift_reports", "outputs", "catboost_logs"]:
    os.makedirs(d, exist_ok=True)

# ─── Model Registry ──────────────────────────────────────────────────
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
    "CatBoost": CatBoostClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

TUNING_SPACE = {
    "LogisticRegression": lambda t: {"C": t.suggest_float("C", 0.01, 10.0)},
    "RandomForest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                               "max_depth": t.suggest_int("max_depth", 3, 15)},
    "XGBoost": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                          "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
                          "max_depth": t.suggest_int("max_depth", 3, 10)},
    "LightGBM": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                           "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2),
                           "num_leaves": t.suggest_int("num_leaves", 20, 150)},
    "CatBoost": lambda t: {"iterations": t.suggest_int("iterations", 50, 200),
                           "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
                           "depth": t.suggest_int("depth", 3, 10)},
    "GradientBoosting": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                                    "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
                                    "max_depth": t.suggest_int("max_depth", 3, 10)},
}

def safe_start_run(name=None, nested=False):
    if mlflow.active_run():
        mlflow.end_run()
    return mlflow.start_run(run_name=name, nested=nested)

def train_all_models(
    df: pd.DataFrame = None,
    data_path: str = "data/processed/13_final_features.csv",
    experiment_name: str = "Lead_Conversion_Modeling",
    target: str = "Converted",
    test_size: float = 0.2,
    timeout: int = 600,
    cv=5,
    n_trials: int = 15
):
    mlflow.set_experiment(experiment_name)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded processed data: {df.shape}")

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    try:
        logger.info("Filtering columns for drift detection to avoid zero-variance error...")
        X_train_numeric = X_train.select_dtypes(include=np.number)
        X_test_numeric = X_test.select_dtypes(include=np.number)
        train_variant_cols = X_train_numeric.columns[X_train_numeric.std() > 0]
        test_variant_cols = X_test_numeric.columns[X_test_numeric.std() > 0]
        safe_numeric_cols = train_variant_cols.intersection(test_variant_cols)
        X_train_drift = pd.concat([X_train[safe_numeric_cols], X_train.select_dtypes(exclude=np.number)], axis=1)
        X_test_drift = pd.concat([X_test[safe_numeric_cols], X_test.select_dtypes(exclude=np.number)], axis=1)
        logger.info(f"Passing {X_train_drift.shape[1]} columns to drift detector.")
        log_drift_report(X_train_drift, X_test_drift, dataset_name="train_vs_test")
    except Exception as e:
        logger.warning(f"⚠️ Could not perform drift detection. Error: {e}")

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = TargetEncoderWrapper(cols_to_encode=cat_cols)
    X_train_enc = encoder.fit_transform(X_train.copy(), y_train)
    X_test_enc = encoder.transform(X_test.copy())

    joblib.dump(encoder, "artifacts/final_target_encoder.joblib")
    logger.info(f"💾 Saved final target encoder to artifacts/final_target_encoder.joblib")

    scaler = StandardScaler().fit(X_train_enc)
    X_train_enc_df = pd.DataFrame(scaler.transform(X_train_enc), columns=X_train_enc.columns)
    X_test_enc_df = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns)
    joblib.dump(scaler, "artifacts/feature_scaler.pkl")

    results, models = [], []
    best_auc, best_model_name, best_run_id = 0.0, None, None
    best_model_object = None # To store the actual best model for SHAP

    with safe_start_run("All_Model_Training") as main_run:
        for name, Cls in CLASSIFIERS.items():
            with mlflow.start_run(run_name=name, nested=True) as nested_run:
                params = optimize_model(Cls, TUNING_SPACE[name], X_train_enc_df, y_train, n_trials=n_trials)
                
                if name == "CatBoost":
                    params['verbose'] = 0
                
                model = Cls(**params)
                model.fit(X_train_enc_df, y_train)
                preds = model.predict(X_test_enc_df)
                probs = model.predict_proba(X_test_enc_df)[:, 1]

                metrics = {
                    "accuracy": accuracy_score(y_test, preds),
                    "precision": precision_score(y_test, preds),
                    "recall": recall_score(y_test, preds),
                    "roc_auc": roc_auc_score(y_test, probs),
                }
                
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                input_example = X_train_enc_df.head(5)
                mlflow.sklearn.log_model(model, name, input_example=input_example)

                if metrics["roc_auc"] > best_auc:
                    best_auc = metrics["roc_auc"]
                    best_model_name = name
                    best_run_id = nested_run.info.run_id
                    best_model_object = model # Save the best model object

                results.append({"model": name, **metrics})
                models.append((name, model))

        with mlflow.start_run(run_name="StackingEnsemble", nested=True) as stack_run:
            stack = StackingClassifier(
                estimators=models,
                final_estimator=LogisticRegression(),
                cv=StratifiedKFold(5),
                n_jobs=1,
                passthrough=True
            )
            
            stack.fit(X_train_enc_df, y_train)
            preds = stack.predict(X_test_enc_df)
            probs = stack.predict_proba(X_test_enc_df)[:, 1]

            stack_metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, probs),
            }
            
            mlflow.log_metrics(stack_metrics)
            
            input_example = X_train_enc_df.head(5)
            mlflow.sklearn.log_model(stack, "StackingEnsemble", input_example=input_example)

            if stack_metrics["roc_auc"] > best_auc:
                best_model_name = "StackingEnsemble"
                best_run_id = stack_run.info.run_id
                best_model_object = stack # Save the best model object

            results.append({"model": "StackingEnsemble", **stack_metrics})

        # --- CHANGE MADE HERE: Enhanced SHAP Explainability Logic ---
        if best_model_object:
            logger.info(f"Calculating SHAP values for the best model: {best_model_name}")
            try:
                explainer = None
                # Use the fast TreeExplainer for supported models
                if isinstance(best_model_object, (RandomForestClassifier, XGBClassifier, LGBMClassifier, GradientBoostingClassifier, CatBoostClassifier)):
                    logger.info("Using shap.TreeExplainer for the best model.")
                    explainer = shap.TreeExplainer(best_model_object, X_train_enc_df)
                    shap_values = explainer.shap_values(X_test_enc_df)
                # Use the flexible KernelExplainer for all other models (like Stacking or Logistic Regression)
                else:
                    logger.info("Using shap.KernelExplainer for the best model.")
                    # Use shap.sample to get a representative background dataset, which is better than .head()
                    background_data = shap.sample(X_train_enc_df, 100)
                    explainer = shap.KernelExplainer(best_model_object.predict_proba, background_data)
                    shap_values = explainer.shap_values(X_test_enc_df)
                
                # For binary classification, SHAP often returns a list [shap_for_class_0, shap_for_class_1]
                # We are interested in the explanation for the positive class (class 1)
                shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

                # Create and save the summary plot
                shap.summary_plot(shap_values_for_plot, X_test_enc_df, plot_type="bar", show=False)
                
                plot_path = "outputs/shap_summary_plot.png"
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                
                # Log the plot as an artifact to the main parent run
                mlflow.log_artifact(plot_path, "explainability")
                logger.info(f"✅ SHAP summary plot saved and logged to MLflow artifacts under 'explainability'.")

            except Exception as e:
                logger.warning(f"⚠️ Could not generate SHAP plot. Error: {e}")
        # --- END OF CHANGE ---

    if best_model_name is not None and best_run_id is not None:
        model_name_in_registry = "LeadConversionModel"
        logger.info(f"Registering best model '{best_model_name}' as '{model_name_in_registry}'")
        
        run_model_uri = f"runs:/{best_run_id}/{best_model_name}"
        mv = mlflow.register_model(run_model_uri, model_name_in_registry)
        logger.info(f"Successfully registered model version {mv.version}.")

        client.set_registered_model_alias(
            name=model_name_in_registry,
            alias="champion",
            version=mv.version
        )
        logger.info(f"Set alias 'champion' for version {mv.version}.")

        for mv_existing in client.search_model_versions(f"name='{model_name_in_registry}'"):
            if mv_existing.current_stage == "Production" and mv_existing.version != mv.version:
                logger.info(f"Archiving old production model version: {mv_existing.version}")
                client.transition_model_version_stage(
                    name=model_name_in_registry,
                    version=mv_existing.version,
                    stage="Archived"
                )

        logger.info(f"Transitioning new model version {mv.version} to 'Production'.")
        client.transition_model_version_stage(
            name=model_name_in_registry,
            version=mv.version,
            stage="Production"
        )
        logger.info("✅ Model registration and promotion complete.")

    df_res = pd.DataFrame(results)
    table = Table(title="Model Comparison")
    table.add_column("Model", style="bold")
    for m in ["accuracy", "precision", "recall", "roc_auc"]:
        table.add_column(m, justify="right")
    for _, r in df_res.iterrows():
        table.add_row(r["model"], *(f"{r[m]:.3f}" for m in ["accuracy", "precision", "recall", "roc_auc"]))
    console.print(table)
    df_res.to_csv("outputs/model_summary.csv", index=False)

    print("\n✅ Final Model Metrics Summary:")
    print(df_res.to_string(index=False))

    return best_model_name
