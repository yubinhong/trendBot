FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py web_app.py run_web.py ./
COPY static/ ./static/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables for Python logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Run the application
CMD ["python", "-u", "main.py"]