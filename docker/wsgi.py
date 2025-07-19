# This file is the entry point for the Gunicorn server.
# It imports the Flask app object from your main app.py file.

from app import app

# The Gunicorn server will look for this 'app' variable.
if __name__ == "__main__":
    app.run()