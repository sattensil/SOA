FROM python:3.10-slim

WORKDIR /app

# Copy poetry configuration files
COPY pyproject.toml ./

# Install poetry
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# Generate lock file and install dependencies
RUN poetry lock && \
    poetry install --without dev --no-root --no-interaction --no-ansi

# Copy application code
COPY . .

# Set environment variables
ENV MODELS_DIR=/app/models
ENV DATA_DIR=/app/data
ENV FEATURES_DIR=/app/features
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "run_api.py"]
