import os
import json
import flask
import traceback
import pandas as pd
import mlflow

from preprocessor import PreprocessingPipeline # Import our preprocessor class

# --- Constants ---
# SageMaker expects the model and artifacts to be in this directory.
MODEL_DIR = "/opt/ml/model/"

# --- Load Model and Preprocessor at Startup ---
# This is done once when the container starts, making predictions faster.
def load_model_and_preprocessor():
    """Load the MLflow model and the preprocessing pipeline."""
    try:
        # The MLflow model is in a subdirectory named after the model's name
        model_name = "StackingEnsemble" # Or the name of your best model
        model_path = os.path.join(MODEL_DIR, model_name)
        model = mlflow.pyfunc.load_model(model_path)
        
        # The preprocessing artifacts are in the 'artifacts' subdirectory
        artifact_path = os.path.join(MODEL_DIR, "artifacts")
        preprocessor = PreprocessingPipeline(artifact_path=artifact_path)
        
        return model, preprocessor
    except Exception as e:
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
            response=json.dumps({"error": "Model or preprocessor not loaded"}),
            status=500,
            mimetype="application/json"
        )

    try:
        # The request data will be in the body of the POST request.
        # We expect JSON format.
        data = flask.request.get_data().decode('utf-8')
        # Convert the JSON string to a pandas DataFrame
        input_df = pd.read_json(data, orient='split')

        # 1. Preprocess the raw data
        processed_df = preprocessor.transform(input_df)

        # 2. Make predictions
        # The MLflow model expects column names to match the training data
        if hasattr(model, 'metadata') and model.metadata.signature:
            model_features = model.metadata.get_input_schema().input_names()
            processed_df = processed_df.reindex(columns=model_features, fill_value=0)
        
        predictions = model.predict(processed_df)

        # 3. Format the response
        # We assume the prediction is the probability of the positive class (1)
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
