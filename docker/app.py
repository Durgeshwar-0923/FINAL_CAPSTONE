import os
import json
import flask
import traceback
import pandas as pd
import numpy as np
import mlflow
import boto3
import uuid
from datetime import datetime

# This preprocessor script is copied into the container and is a local import.
from preprocessor import PreprocessingPipeline 

# --- Constants ---
# When SageMaker deploys the endpoint, it unzips your model.tar.gz package
# into this directory inside the container.
MODEL_DIR = "/opt/ml/model/"

# --- NEW CONFIGURATION: S3 Bucket for Incoming Data ---
# We will store a copy of all incoming prediction requests in this bucket.
# Make sure your SageMaker Execution Role has s3:PutObject permission on this bucket.
INFERENCE_DATA_S3_BUCKET = "flaskcapstonebucket" 

# --- Load Model and Preprocessor at Startup ---
def load_model_and_preprocessor():
    """Load the MLflow model and the preprocessing pipeline from local files."""
    try:
        # The MLflow model files are at the root of the MODEL_DIR.
        model_path = MODEL_DIR
        
        print(f"Loading model from local path: {model_path}")
        model = mlflow.pyfunc.load_model(model_path)
        print("✅ Model loaded successfully.")
        
        # The preprocessing artifacts are in the 'artifacts' subdirectory.
        artifact_path = os.path.join(MODEL_DIR, "artifacts")
        print(f"Loading preprocessor artifacts from: {artifact_path}")
        preprocessor = PreprocessingPipeline(artifact_path=artifact_path)
        print("✅ Preprocessor loaded successfully.")
        
        return model, preprocessor
    except Exception as e:
        print("❌ Error during model or preprocessor loading:")
        traceback.print_exc()
        return None, None

model, preprocessor = load_model_and_preprocessor()

# --- Initialize Flask App ---
app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check endpoint. SageMaker calls this to make sure the container is running.
    """
    status = 200 if model is not None and preprocessor is not None else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def invocations():
    """
    The main prediction endpoint. SageMaker sends prediction requests here.
    """
    if not model or not preprocessor:
        return flask.Response(
            response=json.dumps({"error": "Model or preprocessor not loaded. Check startup logs."}),
            status=500,
            mimetype="application/json"
        )

    try:
        data = flask.request.get_data().decode('utf-8')
        
        # --- ADDITION: Save Incoming Data to S3 ---
        try:
            s3_client = boto3.client("s3")
            # Create a unique, date-partitioned path for the file
            now = datetime.utcnow()
            s3_key = f"incoming-data/year={now.year}/month={now.month:02d}/day={now.day:02d}/{uuid.uuid4()}.json"
            
            s3_client.put_object(
                Bucket=INFERENCE_DATA_S3_BUCKET,
                Key=s3_key,
                Body=data
            )
            print(f"✅ Successfully saved incoming data to s3://{INFERENCE_DATA_S3_BUCKET}/{s3_key}")
        except Exception as s3_e:
            # Log the error but do not stop the prediction
            print(f"⚠️ WARNING: Failed to save incoming data to S3. Error: {s3_e}")
        # --- END OF ADDITION ---
        
        input_df = pd.read_json(data, orient='split')

        # 1. Preprocess the raw data
        processed_df = preprocessor.transform(input_df)

        # 2. Make predictions
        if hasattr(model, 'metadata') and model.metadata.signature:
            model_features = model.metadata.get_input_schema().input_names()
            processed_df = processed_df.reindex(columns=model_features, fill_value=0)
        
        predictions = model.predict(processed_df)

        # 3. Format the response
        if isinstance(predictions, pd.DataFrame):
            probs = predictions.iloc[:, 0].tolist()
        elif isinstance(predictions, np.ndarray):
            probs = predictions.tolist()
        else:
            probs = [float(p) for p in predictions]

        result = {
            "probabilities": probs
        }
        
        return flask.Response(response=json.dumps(result), status=200, mimetype="application/json")

    except Exception as e:
        tb = traceback.format_exc()
        return flask.Response(
            response=json.dumps({"error": str(e), "traceback": tb}),
            status=400,
            mimetype="application/json"
        )
