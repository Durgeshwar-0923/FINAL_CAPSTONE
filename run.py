import os
import pandas as pd
import mlflow
from datetime import datetime

# ---------- 1. SageMaker‑hosted MLflow setup ----------
MLFLOW_TRACKING_ARN = "arn:aws:sagemaker:ap-south-1:439932142398:mlflow-tracking-server/sagemaker"
MLFLOW_S3_BUCKET    = "s3://mlflowminfycapstone/Artifacts/"
MLFLOW_S3_ENDPOINT  = "https://s3.ap-south-1.amazonaws.com"

# Set environment variables for SageMaker-hosted MLflow
mlflow.set_tracking_uri("https://t-egcu1qkoqhtg.ap-south-1.experiments.sagemaker.aws")
os.environ["MLFLOW_TRACKING_URI"]        = MLFLOW_TRACKING_ARN
os.environ["MLFLOW_S3_ENDPOINT_URL"]     = MLFLOW_S3_ENDPOINT
os.environ["MLFLOW_EXPERIMENT_BUCKET"]   = MLFLOW_S3_BUCKET

print("✅ MLflow Tracking URI set to:", os.environ["MLFLOW_TRACKING_URI"])

# ---------- 2. Imports ----------
from src.config.config import Paths, RedshiftConfig
from src.data_ingestion.data_loader import DataLoader
from src.data_processing.preprocessor.pipeline import run_pipeline_with_tracking
from src.data_processing.eda import run_sweetviz, eda_summary
from src.models.train_models import train_all_models
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------- 3. Main Pipeline ----------
def main():
    """
    Main orchestrator for the entire Lead Conversion ML Pipeline.
    """
    paths = Paths()

    # Experiment and Target Configs
    EDA_EXPERIMENT_NAME      = "Lead_Conversion_EDA"
    TRAINING_EXPERIMENT_NAME = "Lead_Conversion_Modeling"
    TARGET_COLUMN            = "converted"

    # Ensure required directories exist
    paths.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    paths.REPORTS.mkdir(parents=True, exist_ok=True)

    # ---------- 4. Data Loading from Redshift (Optional) ----------
    # logger.info("🔄 Connecting to Redshift and loading raw data...")
    # loader = DataLoader(RedshiftConfig())
    # df_raw = loader.load_data()
    # staged_path = paths.RAW_DATA / "staged_lead_scoring_data.csv"
    # df_raw.to_csv(staged_path, index=False)

    # ---------- 5. EDA (Optional) ----------
    # mlflow.set_experiment(EDA_EXPERIMENT_NAME)
    # with mlflow.start_run(run_name="EDA_Report"):
    #     eda_summary(df_raw, target_col=TARGET_COLUMN)
    #     eda_file = paths.REPORTS / "eda_report.html"
    #     run_sweetviz(df_raw, TARGET_COLUMN, output_path=str(eda_file))
    #     mlflow.log_artifact(str(eda_file), artifact_path="EDA")

    # ---------- 6. Preprocessing (Optional) ----------
    # logger.info("🛠️ Running preprocessing pipeline...")
    # run_pipeline_with_tracking(raw_data_path=str(staged_path), is_training=True)

    # ---------- 7. Model Training ----------
    final_path = paths.PROCESSED_DATA / "13_final_features.csv"
    if not final_path.exists():
        logger.error(f"❌ Missing preprocessed file: {final_path}")
        return

    processed_df = pd.read_csv(final_path)
    mlflow.set_experiment(TRAINING_EXPERIMENT_NAME)

    logger.info("🚀 Starting model training...")
    best_model = train_all_models(
        df=processed_df,
        target=TARGET_COLUMN,
        experiment_name=TRAINING_EXPERIMENT_NAME,
        n_trials=15,
        timeout=900,
        cv=5
    )
    logger.info(f"✅ Training complete. Best model: {best_model}")

# ---------- 8. Entry Point ----------
if __name__ == "__main__":
    main()
