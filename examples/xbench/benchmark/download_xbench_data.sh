#!/bin/bash

# XBench Dataset Download and Decrypt Script
# This script downloads the official xbench-evals dataset and decrypts it for local use

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/temp_xbench"
REPO_URL="https://github.com/xbench-ai/xbench-evals.git"

echo "🚀 Starting XBench dataset download and decryption..."
echo ""

# Step 1: Clone the repository
echo "📦 Step 1/4: Cloning xbench-evals repository..."
if [ -d "$TEMP_DIR" ]; then
    echo "   ⚠️  Temporary directory already exists, removing..."
    rm -rf "$TEMP_DIR"
fi

git clone "$REPO_URL" "$TEMP_DIR"
cd "$TEMP_DIR"

echo "   ✅ Repository cloned successfully"
echo ""

# Step 2: Install dependencies
echo "📦 Step 2/4: Installing dependencies..."
pip install -q pandas
echo "   ✅ Dependencies installed"
echo ""

# Step 3: Decrypt dataset
echo "🔓 Step 3/4: Decrypting DeepSearch dataset..."
echo "   This may take a moment..."

echo "$SCRIPT_DIR"
# Find and decrypt all DeepSearch CSV files
DEEPSEARCH_FILES=($(find "$TEMP_DIR/data" -name "DeepSearch*.csv" ! -name "*_decrypted.csv" | sort))
TOTAL_FILES=${#DEEPSEARCH_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    echo "   ⚠️  No DeepSearch files found!"
else
    echo "   📊 Found $TOTAL_FILES DeepSearch file(s) to decrypt"
    
    for i in "${!DEEPSEARCH_FILES[@]}"; do
        FILE="${DEEPSEARCH_FILES[$i]}"
        FILENAME=$(basename "$FILE")
        OUTPUT_FILE="${FILE%.csv}_decrypted.csv"
        
        echo "   🔐 [$((i+1))/$TOTAL_FILES] Decrypting $FILENAME..."
        python "$SCRIPT_DIR/decrypt_xbench.py" "$FILE" "$OUTPUT_FILE"
    done
    
    echo "   ✅ All datasets decrypted"
fi
echo ""

# Step 4: Copy to benchmark directory
echo "📋 Step 4/4: Copying decrypted dataset..."
DECRYPTED_FILES=($(find "$TEMP_DIR/data" -name "DeepSearch*_decrypted.csv" | sort))
COPIED_COUNT=0

if [ ${#DECRYPTED_FILES[@]} -eq 0 ]; then
    echo "   ⚠️  No decrypted files found to copy!"
else
    for FILE in "${DECRYPTED_FILES[@]}"; do
        FILENAME=$(basename "$FILE")
        cp "$FILE" "$SCRIPT_DIR/"
        echo "   ✅ $FILENAME copied"
        ((COPIED_COUNT++))
    done
    echo "   📊 Total files copied: $COPIED_COUNT"
fi

echo ""

# Step 5: Cleanup
echo "🧹 Cleaning up temporary files..."
cd "$SCRIPT_DIR"
rm -rf "$TEMP_DIR"
echo "   ✅ Cleanup complete"
echo ""

# Final message
echo "✅ XBench dataset download and decryption completed successfully!"
echo ""
echo "📁 Downloaded files in: $SCRIPT_DIR"
echo ""
echo "⚠️  IMPORTANT SECURITY NOTICE:"
echo "   - DO NOT upload the decrypted data online"
echo "   - DO NOT commit decrypted data to public repositories"
echo "   - Keep the decrypted datasets local only"
echo ""
echo "🌐 For more information, visit: https://xbench.org"
echo ""

