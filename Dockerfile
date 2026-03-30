# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (this automatically grabs your new 'server' folder)
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the Uvicorn server pointing to the new server/app.py file
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]