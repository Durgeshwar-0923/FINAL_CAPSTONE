import pandas as pd
from sqlalchemy import create_engine, exc
import os
from src.config.config import RedshiftConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_csv_to_redshift(
    csv_path: str = os.path.join("data", "raw", "Lead Scoring.csv"),
    table_name: str = None,
    if_exists_policy: str = "replace",
    chunksize: int = 1000
) -> None:
    """
    Loads data from a CSV file into an Amazon Redshift table in chunks.

    Args:
        csv_path (str): The path to the CSV file.
        table_name (str): The name of the target Redshift table.
        if_exists_policy (str): How to behave if the table already exists.
                                Options: 'replace', 'append', 'fail'.
        chunksize (int): The number of rows to write in each batch.
    """
    try:
        config = RedshiftConfig()
        if table_name is None:
            table_name = config.table_name

        # --- ✅ Redshift: Create SQLAlchemy engine for Redshift ---
        # This enables pandas `.to_sql()` to load data into Redshift.
        # Make sure `config.db_url` is your Redshift-compatible SQLAlchemy URI.
        engine = create_engine(config.db_url)

        logger.info(f"📤 Loading CSV: {csv_path} → Redshift table: {table_name}")

        # --- 🔄 Efficient chunked loading for large CSVs ---
        # Avoids memory issues and allows partial retries if needed.
        csv_iterator = pd.read_csv(csv_path, iterator=True, chunksize=chunksize)

        for i, chunk in enumerate(csv_iterator):
            policy = if_exists_policy if i == 0 else "append"
            chunk.to_sql(table_name, engine, if_exists=policy, index=False)
            logger.info(f"📝 Wrote chunk {i+1} to table {table_name}")

        logger.info(f"✅ Data from {csv_path} successfully loaded into Redshift.")

    except FileNotFoundError:
        logger.error(f"❌ File not found at path: {csv_path}")
    except exc.SQLAlchemyError as e:
        logger.error(f"❌ A database error occurred: {e}")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")

    # --- 🗒️ COMMENTED: PostgreSQL Logic Block ---
    # Uncomment the following if you want to revert to PostgreSQL instead of Redshift
    #
    # try:
    #     config = DatabaseConfig()
    #     if table_name is None:
    #         table_name = config.table_name
    #
    #     config.ensure_database_exists()
    #     engine = create_engine(config.db_url)
    #     logger.info(f"📤 Loading CSV: {csv_path} → PostgreSQL table: {table_name}")
    #
    #     csv_iterator = pd.read_csv(csv_path, iterator=True, chunksize=chunksize)
    #     for i, chunk in enumerate(csv_iterator):
    #         policy = if_exists_policy if i == 0 else "append"
    #         chunk.to_sql(table_name, engine, if_exists=policy, index=False)
    #         logger.info(f"📝 Wrote chunk {i+1} to table {table_name}")
    #
    #     logger.info(f"✅ Data from {csv_path} loaded into PostgreSQL successfully.")
    #
    # except Exception as e:
    #     logger.error(f"❌ An error occurred during PostgreSQL load: {e}")


# 🧪 Entry point for testing or CLI run
if __name__ == "__main__":
    load_csv_to_redshift(if_exists_policy="replace")
