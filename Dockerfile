# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- FIX APPLIED HERE ---
# Install system dependencies required for the application
# Added 'libgomp1', which provides the missing libgomp.so.1 library required by LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# --- END OF FIX ---

# Copy the deployment-specific requirements file (flask.txt) into the container
# This works because both Dockerfile and flask.txt are in the project root.
COPY flask.txt /

# Install Python packages from your flask.txt file
RUN pip install --no-cache-dir -r /flask.txt

# Set the working directory inside the container
WORKDIR /opt/program

# Copy the Nginx configuration file into the container from the 'docker' subfolder
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Copy the Flask application, preprocessor, and all source code
# The paths are now correct because the Dockerfile is in the root.
COPY docker/app.py docker/wsgi.py docker/preprocessor.py /opt/program/
COPY src /opt/program/src

# Copy the serve script and make it executable
COPY docker/serve /opt/program/
RUN chmod +x /opt/program/serve

# Expose the port that Nginx will listen on
EXPOSE 8080

# Define the command to run when the container starts
ENTRYPOINT ["/opt/program/serve"]
