# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for the application
# - nginx: The web server that SageMaker requires
# - ca-certificates: For secure HTTPS connections
# - gunicorn & gevent: High-performance Python web server gateway interface
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages required by the application
# We use a multi-stage approach to keep the final image clean
COPY ../requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the Nginx configuration file into the container
COPY nginx.conf /etc/nginx/nginx.conf

# Set the working directory inside the container
WORKDIR /opt/program

# Copy the Flask application and preprocessing code into the container
# We copy the entire 'src' directory and the docker-specific files
COPY ../src /opt/program/src
COPY app.py wsgi.py preprocessor.py /opt/program/

# Copy the serve script and make it executable
COPY serve /opt/program/
RUN chmod +x /opt/program/serve

# Expose the port that Nginx will listen on
EXPOSE 8080

# Define the command to run when the container starts
# SageMaker will execute this 'serve' script
ENTRYPOINT ["/opt/program/serve"]
