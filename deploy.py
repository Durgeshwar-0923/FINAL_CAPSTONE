import sagemaker
import boto3
import os
import subprocess

# --- Configuration ---
# 1. This is the name of the ECR repository that sm-docker uses by default.
IMAGE_NAME = "sagemaker-studio"
# We can give our endpoint a more descriptive name.
ENDPOINT_NAME = "lead-conversion-predictor"

# 2. This is the S3 URI of your model.tar.gz file.
model_s3_uri = "s3://flaskcapstonebucket/models/LeadConversionModel/model.tar.gz"

# 3. Get the SageMaker execution role.
try:
    role = sagemaker.get_execution_role()
    print(f"SageMaker Execution Role ARN: {role}")
except Exception:
    print("Could not get execution role automatically. Please set it manually if needed.")
    # role = "arn:aws:iam::..."

# --- Sanity Check ---
if "PASTE_YOUR_S3_URI_HERE" in model_s3_uri:
    raise ValueError("Please update the 'model_s3_uri' variable with the correct S3 path.")

# --- Step 1: Get AWS Account Info & ECR Repository URI ---
print("\n--- Step 1: Configuring ECR Repository ---")
try:
    sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    
    session = boto3.session.Session()
    region = session.region_name
    
    # This is the full, correct URI that the image will have in ECR
    ecr_repository_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{IMAGE_NAME}"
    print(f"ECR Repository URI: {ecr_repository_uri}")
    
    # The ECR repository was already created in the previous run.
    print(f"Assuming ECR repository '{IMAGE_NAME}' already exists.")

except Exception as e:
    print(f"❌ Failed to configure ECR. Error: {e}")
    raise e

# --- Step 2: Build and Push the Docker Image using the sm-docker CLI ---
print("\n--- Step 2: Building and Pushing Docker Image to ECR ---")
# This step MUST be run again to build the container with the corrected Dockerfile.

try:
    build_command = f"sm-docker build . --repository {IMAGE_NAME}:latest"
    
    print(f"\nRunning build command: {build_command}")
    print("This will take 10-15 minutes and will show a lot of log output...")
    
    process = subprocess.Popen(build_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    rc = process.poll()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, build_command)
    
    print("\n✅ Successfully built and pushed image to ECR.")

except subprocess.CalledProcessError as e:
    print(f"❌ Docker build and push failed with return code {e.returncode}.")
    print("Please check the build logs above for the specific error.")
    raise e
except Exception as e:
    print(f"❌ An unexpected error occurred during the build process: {e}")
    raise e


# --- Step 3: Create a SageMaker Model ---
print("\n--- Step 3: Creating SageMaker Model ---")
from sagemaker.model import Model

try:
    sagemaker_session = sagemaker.Session()
    
    model = Model(
        image_uri=ecr_repository_uri,      # Use the full, correct ECR URI
        model_data=model_s3_uri,
        role=role,
        sagemaker_session=sagemaker_session,
        name=ENDPOINT_NAME 
    )
    print(f"✅ SageMaker Model resource '{ENDPOINT_NAME}' created.")

except Exception as e:
    print(f"❌ Failed to create SageMaker Model. Error: {e}")
    raise e

# --- Step 4: Deploy the Model to a Real-Time Endpoint ---
print("\n--- Step 4: Deploying Model to a Real-Time Endpoint ---")
print("This step will take several minutes as SageMaker provisions the infrastructure...")

try:
    # Clean up any old endpoints with the same name before deploying
    try:
        sagemaker_session.delete_endpoint(endpoint_name=ENDPOINT_NAME)
        sagemaker_session.delete_endpoint_config(endpoint_config_name=ENDPOINT_NAME)
        print("Cleaned up existing endpoint and config with the same name.")
    except Exception:
        print("No existing endpoint to clean up.")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large', # Using your specified instance type
        endpoint_name=ENDPOINT_NAME
    )
    print(f"\n🎉 Successfully deployed endpoint '{ENDPOINT_NAME}'!")
    print(f"Endpoint Name: {predictor.endpoint_name}")

except Exception as e:
    print(f"❌ Endpoint deployment failed. Error: {e}")
    try:
        sagemaker_session.delete_endpoint(endpoint_name=ENDPOINT_NAME)
        sagemaker_session.delete_endpoint_config(endpoint_config_name=ENDPOINT_NAME)
        model.delete_model()
        print("Cleaned up SageMaker resources after failure.")
    except Exception as cleanup_e:
        print(f"Error during cleanup: {cleanup_e}")
    raise e
