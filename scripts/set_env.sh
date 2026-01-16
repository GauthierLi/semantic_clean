#!/bin/bash

# Set environment variables for the project
# This script should be sourced before running any Python scripts
# Usage: source setup_env.sh

# Get the project root directory
# This works whether the script is sourced or executed
if [ -n "${BASH_SOURCE[0]}" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
    PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

# Add proto_gen to Python path
export PYTHONPATH="${PROJECT_ROOT}/proto_gen:${PYTHONPATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "✅ Environment variables set:"
echo "   PROJECT_ROOT: $PROJECT_ROOT"
echo "   PYTHONPATH: $PYTHONPATH"