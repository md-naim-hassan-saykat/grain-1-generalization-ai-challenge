#!/bin/bash
# Script to create a properly structured zip file for Codabench
# This script creates a zip with competition.yaml at the root level

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUNDLE_NAME="grain_classification_bundle.zip"

# Remove old zip if it exists
if [ -f "$SCRIPT_DIR/$BUNDLE_NAME" ]; then
    echo "Removing old bundle: $BUNDLE_NAME"
    rm "$SCRIPT_DIR/$BUNDLE_NAME"
fi

# Create zip from inside Competition_Bundle directory
# This ensures competition.yaml is at the root of the zip
cd "$SCRIPT_DIR"
zip -r "$BUNDLE_NAME" . -x "*.git*" "*.ipynb_checkpoints*" "*.DS_Store" "preparation.ipynb" "STATUS.md" "utilities/*" "data_split_summary.json"

echo ""
echo "✓ Bundle created: $BUNDLE_NAME"
echo "✓ competition.yaml should be at the root of the zip"
echo ""
echo "To verify, run: unzip -l $BUNDLE_NAME | grep competition.yaml"
echo "Expected output: competition.yaml (not Competition_Bundle/competition.yaml)"
