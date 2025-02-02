# Use official Python base image matching Pipfile's Python version
FROM python:3.12-slim

# Install system dependencies required for PyTorch and image processing
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libjpeg-dev \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && \
    pipenv requirements > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application files
COPY app/ ./app/

# Create directory for uploads and set permissions
RUN mkdir -p /app/uploads && \
    chmod 777 /app/uploads

# Environment variables
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production
ENV UPLOAD_FOLDER=/app/uploads

# Expose the Flask port
EXPOSE 5000

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
