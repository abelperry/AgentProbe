#!/bin/bash
set -euo pipefail

# Check if we're in a valid working directory
if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."
    exit 1
fi

PYTHON_BIN=""
if command -v python &> /dev/null; then
    PYTHON_BIN="python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    if ! command -v apt-get &> /dev/null; then
        echo "Error: python not available and apt-get missing"
        echo "0" > /logs/verifier/reward.txt
        exit 1
    fi
    apt-get update -y > /dev/null 2>&1
    apt-get install -y python3 > /dev/null 2>&1
    PYTHON_BIN="python3"
fi

OUTPUT=$($PYTHON_BIN /tests/test_outputs.py 2>&1)
STATUS=$?

echo "$OUTPUT"
mkdir -p /logs/verifier

if [ $STATUS -ne 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

TOTAL=$(echo "$OUTPUT" | awk '/^ac[0-9]+:/{count++} END{print count+0}')
PASSED=$(echo "$OUTPUT" | awk '/^ac[0-9]+:满足$/{count++} END{print count+0}')

if [ -z "$TOTAL" ] || [ "$TOTAL" -eq 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

if [ "$PASSED" -eq "$TOTAL" ]; then
    echo "1" > /logs/verifier/reward.txt
    exit 0
else
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi
