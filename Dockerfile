FROM python:3.11-slim

# Install ffmpeg and system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Create upload/output dirs
RUN mkdir -p backend/uploads backend/outputs

# Expose port
EXPOSE 8000

# Start gunicorn
CMD cd backend && gunicorn --workers 1 --timeout 7200 --worker-class sync --bind 0.0.0.0:${PORT:-8000} app:app
