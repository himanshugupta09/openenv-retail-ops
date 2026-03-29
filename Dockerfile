# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces requires the app to listen on 0.0.0.0:7860
# For a pure scripting environment, you usually CMD the inference script, 
# but if the OpenEnv validator requires a running server:
# CMD ["python", "inference.py"]
# Or if it needs a dummy FastAPI server to pass the HF deployment ping:
RUN pip install fastapi uvicorn
COPY server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]