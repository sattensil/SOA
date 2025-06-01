#!/bin/bash
# Script to run all unit tests

# Navigate to the project root directory where pyproject.toml is located
cd "$(dirname "$0")"
cd ..

# Run all unit tests using Poetry
echo "Running unit tests..."
poetry run python -m unittest discover -s advanced_implementation/tests/unit

# Optional: Run with coverage
if [ "$1" == "--coverage" ]; then
    echo "Running tests with coverage..."
    poetry run coverage run -m unittest discover -s advanced_implementation/tests/unit
    poetry run coverage report -m
    poetry run coverage html
    echo "Coverage report generated in htmlcov/"
fi

echo "Done!"
