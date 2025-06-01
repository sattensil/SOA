#!/bin/bash

# Run the enhanced Mine Safety Injury Rate Prediction pipeline
# This script uses Poetry to manage dependencies

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Navigate to the project directory
cd "$(dirname "$0")"

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Installing dependencies with Poetry..."
    poetry install
fi

# Run the enhanced pipeline with Poetry
echo "Running enhanced Mine Safety Injury Rate Prediction pipeline..."
poetry run python scripts/enhanced_main.py "$@"

echo "Enhanced pipeline completed."
