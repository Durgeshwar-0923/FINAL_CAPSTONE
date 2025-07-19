import os
import json
import boto3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# -------------------------------
# ✅ Path Configurations
# -------------------------------
@dataclass
class Paths:
    ROOT: Path = Path(__file__).parent.parent.parent
    DATA: Path = ROOT / "data"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"
    RAW_DATA: Path = DATA / "raw"
    PROCESSED_DATA: Path = DATA / "processed"
    DRIFT_REPORTS: Path = DATA / "drift_reports"

# -------------------------------
# ✅ Redshift Configuration (Active)
# -------------------------------
@dataclass
class RedshiftConfig:
    secret_name: str = os.getenv("REDSHIFT_SECRET_NAME", "")
    region_name: str = os.getenv("AWS_REGION", "ap-south-1")
    cluster_id: str = os.getenv("REDSHIFT_CLUSTER_ID", "")
    database: str = os.getenv("REDSHIFT_DB_NAME", "dev")
    schema_name: str = os.getenv("REDSHIFT_SCHEMA", "public")
    table_name: str = os.getenv("REDSHIFT_TABLE", "lead_scoring")

    def get_credentials(self) -> dict:
        print("🔐 Connecting to AWS Secrets Manager...")
        client = boto3.client('secretsmanager', region_name=self.region_name)
        try:
            get_secret_value_response = client.get_secret_value(SecretId=self.secret_name)
            secret_dict = json.loads(get_secret_value_response['SecretString'])

            required_keys = ['host', 'port', 'username', 'password']
            for key in required_keys:
                if key not in secret_dict:
                    raise KeyError(f"❌ Missing required key '{key}' in secret.")

            print("✅ Secret retrieved successfully from Secrets Manager.")
            return secret_dict

        except Exception as e:
            print(f"❌ Failed to retrieve secret: {e}")
            raise

    @property
    def db_url(self) -> str:
        creds = self.get_credentials()
        password_safe = quote_plus(creds['password'])
        return (
            f"redshift+psycopg2://{creds['username']}:{password_safe}"
            f"@{creds['host']}:{creds['port']}/{self.database}"
        )

# -------------------------------
# ✅ PostgreSQL (Retained but Commented)
# -------------------------------
"""
@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 5432))
    database: str = os.getenv("DB_NAME", "Lead_db")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASS", "Minfy")
    schema_name: str = "public"
    table_name: str = "Lead_data"

    @property
    def db_url(self) -> str:
        password_safe = quote_plus(self.password)
        return f"postgresql://{self.user}:{password_safe}@{self.host}:{self.port}/{self.database}"
"""

# -------------------------------
# ✅ ML Experiment Settings
# -------------------------------
@dataclass
class MLConfig:
    target_column: str = "converted"
    random_state: int = 42
    test_size: float = 0.2

# -------------------------------
# ✅ MLOps Settings (SageMaker MLflow)
# -------------------------------
@dataclass
class MLOpsConfig:
    # ✅ Set to the SageMaker-hosted MLflow Tracking URI (ARN format for SageMaker)
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://t-ifcgkwdbncdu.ap-south-1.experiments.sagemaker.aws"  # Replace with actual URL if needed
    )
    
    # ✅ Artifact storage location in S3
    mlflow_artifact_uri: str = os.getenv(
        "MLFLOW_ARTIFACT_URI",
        "s3://mlflow-capstone-artifacts/ML_FLOW_EXPERIMENTS/"
    )

    # ✅ Optional DAG folder path for Airflow
    airflow_dag_folder: Path = Paths.ROOT / "airflow" / "dags"
