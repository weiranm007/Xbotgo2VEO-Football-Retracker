FROM python:3.11-slim

# Install ffmpeg and all OpenCV system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

RUN mkdir -p backend/uploads backend/outputs

# Smoke test — will show exact import error if any
RUN cd backend && python -c "import app; print('startup OK')"

EXPOSE 8000
CMD cd backend && gunicorn --workers 1 --timeout 7200 --worker-class sync --bind 0.0.0.0:${PORT:-8000} app:app
