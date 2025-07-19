import os
import json
import pandas as pd
import boto3
from sqlalchemy import create_engine
from src.config.config import RedshiftConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Handles loading data from Redshift using AWS Secrets Manager."""

    def __init__(self, config: RedshiftConfig):
        self.config = config
        self.engine = None

        # Required environment vars for Redshift connection
        secret_name = os.getenv("REDSHIFT_SECRET_NAME")
        region      = os.getenv("AWS_REGION")
        db_name     = os.getenv("REDSHIFT_DB_NAME")

        if not (secret_name and region and db_name):
            logger.error("⚠️ Missing one of REDSHIFT_SECRET_NAME, AWS_REGION, or REDSHIFT_DB_NAME in environment.")
            raise EnvironmentError("Redshift environment variables not fully set.")

        logger.info("🔄 Attempting Redshift connection using AWS Secrets Manager...")
        try:
            # Retrieve secret from AWS Secrets Manager
            client = boto3.client("secretsmanager", region_name=region)
            resp   = client.get_secret_value(SecretId=secret_name)
            creds  = json.loads(resp['SecretString'])

            # Extract credentials with multiple key fallbacks
            user     = (creds.get("username")   or creds.get("USERNAME")
                        or creds.get("dbUser")   or creds.get("DBUSER"))
            password = (creds.get("password")   or creds.get("PASSWORD")
                        or creds.get("dbPassword") or creds.get("DBPASSWORD"))
            host     = (creds.get("host")       or creds.get("HOST")
                        or creds.get("dbHost")   or creds.get("DBHOST")
                        or os.getenv("REDSHIFT_HOST"))
            port_raw = (creds.get("port")       or creds.get("PORT")
                        or creds.get("dbPort")  or creds.get("DBPORT")
                        or os.getenv("REDSHIFT_PORT", 5439))

            # Cast port to int, default to 5439 on error
            try:
                port = int(port_raw)
            except (TypeError, ValueError):
                port = 5439

            # Construct Redshift SQLAlchemy URL
            url = (
                f"postgresql+psycopg2://{user}:{password}@"
                f"{host}:{port}/{db_name}"
            )
            self.engine = create_engine(url)
            logger.info("✅ Successfully created Redshift engine via Secrets Manager.")

        except Exception as e:
            logger.error(f"❌ Failed to connect to Redshift: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified table in Redshift via the SQLAlchemy engine.
        """
        schema = self.config.schema_name
        table  = self.config.table_name
        query  = f'SELECT * FROM "{schema}"."{table}"'

        logger.info(f"📦 Executing query: {query}")
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            logger.info(f"✅ Successfully loaded {len(df)} rows from {schema}.{table}.")
            return df
        except Exception as e:
            logger.error(f"❌ Error executing SQL query: {e}")
            raise

# --- PostgreSQL fallback logic retained but commented out ---
#    else:
#        # Uncomment and configure to use PostgreSQL instead of Redshift
#        try:
#            self.engine = create_engine(config.db_url)
#            logger.info("✅ PostgreSQL engine created successfully.")
#        except Exception as e:
#            logger.error(f"❌ Failed to create PostgreSQL engine: {e}")
#            raise
