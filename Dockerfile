# Use official lightweight Python 3.11 environment
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install build essential libraries required for compiling numerical extensions (numpy, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build backend tool (hatchling)
RUN pip install --no-cache-dir --upgrade pip hatchling

# Copy project metadata first to leverage Docker layer caching for dependencies
COPY pyproject.toml README.md ./

# Copy the rest of the application files
COPY . .

# Install the project and all dependencies defined in pyproject.toml
RUN pip install --no-cache-dir .

# Expose port 8000 for FastAPI / Uvicorn
EXPOSE 8000

# Start Uvicorn server bound to 0.0.0.0 to accept external requests
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
