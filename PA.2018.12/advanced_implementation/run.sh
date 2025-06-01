#!/bin/bash

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Install dependencies if needed
if ! poetry check; then
    echo "Setting up Poetry environment..."
    poetry install
fi

# Run the main script with Poetry
echo "Running main.py with Poetry..."
poetry run python -m scripts.main "$@"
