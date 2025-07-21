"""
Module: src/api/app.py
Description: This is the FRONT-END Flask application. It serves the user interface,
             saves incoming data to S3, calls the deployed SageMaker endpoint
             to get predictions, and displays the results.
"""
import os
import uuid
import pandas as pd
import flask
import boto3
import json
from datetime import datetime

from werkzeug.utils import secure_filename
from src.utils.logger import setup_logger

# --- Initial Setup ---
logger = setup_logger(__name__)
app = flask.Flask(__name__, template_folder='templates')

# --- Configuration ---
# This is the name of the live endpoint you deployed.
ENDPOINT_NAME = "lead-conversion-predictor"
# This should be the same region where you deployed your endpoint.
AWS_REGION = "ap-south-1" 
# --- NEW: S3 Bucket for Incoming Data ---
# We will store a copy of all incoming prediction requests in this bucket.
INFERENCE_DATA_S3_BUCKET = "flaskcapstonebucket" 

# --- Path Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
PRED_FOLDER = os.path.join(PROJECT_ROOT, "predictions")
REFERENCE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Lead Scoring.csv")

MODEL_NAME = os.getenv("MODEL_NAME", "LeadConversionModel")
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a_default_secret_key_for_development")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)


@app.route("/")
def home():
    """Renders the main landing page (index.html)."""
    return flask.render_template("index.html", model_name=MODEL_NAME, model_loaded=True)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles the file upload, saves data to S3, calls the SageMaker endpoint,
    and displays the results (result.html).
    """
    if 'file' not in flask.request.files:
        flask.flash("No file part in the request.", "warning")
        return flask.redirect(flask.url_for('home'))

    file = flask.request.files['file']
    if file.filename == '':
        flask.flash("No file selected for uploading.", "warning")
        return flask.redirect(flask.url_for('home'))

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        
        try:
            # Read the file content into memory to use it multiple times
            file_content = file.read()
            # Reset the file stream position so pandas can read it
            file.stream.seek(0) 
            
            # --- ADDITION: Save Incoming Data to S3 ---
            try:
                s3_client = boto3.client("s3")
                # Create a unique, date-partitioned path for the file
                now = datetime.utcnow()
                s3_key = f"incoming-data/year={now.year}/month={now.month:02d}/day={now.day:02d}/{uuid.uuid4()}_{filename}"
                
                s3_client.put_object(
                    Bucket=INFERENCE_DATA_S3_BUCKET,
                    Key=s3_key,
                    Body=file_content
                )
                logger.info(f"✅ Successfully saved incoming data to s3://{INFERENCE_DATA_S3_BUCKET}/{s3_key}")
            except Exception as s3_e:
                # Log the error but do not stop the prediction
                logger.warning(f"⚠️ WARNING: Failed to save incoming data to S3. Error: {s3_e}")
            # --- END OF ADDITION ---

            raw_df = pd.read_csv(file.stream)
            logger.info(f"Read {len(raw_df)} rows from the uploaded file.")
            
            # Convert the DataFrame to the JSON format the endpoint expects
            payload = raw_df.to_json(orient="split")
            
            # Create a SageMaker client and invoke the endpoint
            logger.info(f"Invoking SageMaker endpoint: {ENDPOINT_NAME}")
            sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=payload
            )
            
            # Parse the JSON response from the endpoint
            response_body = response['Body'].read().decode('utf-8')
            result = json.loads(response_body)
            probabilities = result.get("probabilities", [])

            # Add the results back to your original DataFrame for display
            results_df = raw_df.copy()
            results_df['Lead_Conversion_Probability'] = [f"{p:.2%}" for p in probabilities]
            results_df['Lead_Converted_Prediction'] = (pd.Series(probabilities) > 0.5).astype(int)
            logger.info("Successfully received and formatted predictions from SageMaker.")

            # Save the final results to a local file
            result_id = uuid.uuid4().hex[:8]
            result_file = f"prediction_{result_id}_{filename}"
            result_path = os.path.join(PRED_FOLDER, result_file)
            results_df.to_csv(result_path, index=False)
            logger.info(f"Saved prediction results to '{result_path}'")

            # Prepare data for rendering in the HTML template
            table_headers = results_df.columns.tolist()
            table_rows = results_df.head(100).values.tolist()
            
            return flask.render_template(
                "result.html",
                table_headers=table_headers,
                table_rows=table_rows,
                result_file=result_file,
                model_name=MODEL_NAME,
                num_records=len(results_df)
            )

        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            flask.flash(f"An error occurred during processing: {e}", "danger")
            return flask.redirect(flask.url_for('home'))
    else:
        flask.flash("Invalid file type. Please upload a CSV file.", "danger")
        return flask.redirect(flask.url_for('home'))

@app.route("/sample")
def sample():
    logger.info("Providing sample data file for download.")
    return flask.send_file(REFERENCE_DATA_PATH, as_attachment=True)


@app.route("/download/<filename>")
def download(filename):
    logger.info(f"Processing download request for file: {filename}")
    safe_filename = secure_filename(filename)
    file_path = os.path.join(PRED_FOLDER, safe_filename)
    if os.path.exists(file_path):
        return flask.send_file(file_path, as_attachment=True)
    else:
        flask.flash("File not found.", "danger")
        return flask.redirect(flask.url_for('home'))


if __name__ == "__main__":
    # This runs your local front-end application
    app.run(debug=True, port=int(os.getenv("PORT", 8000)))
