#!/bin/bash

# Run integration tests for the enhanced mine safety injury rate prediction pipeline
echo "Running integration tests..."
cd "$(dirname "$0")"
poetry run python -m unittest discover -s tests/integration -v

# Check if tests passed
if [ $? -eq 0 ]; then
    echo "✅ All integration tests passed!"
    exit 0
else
    echo "❌ Some integration tests failed."
    exit 1
fi
