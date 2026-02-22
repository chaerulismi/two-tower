#!/usr/bin/env bash
# Downloads and extracts the MovieLens 1M dataset into data/raw/
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH="$DATA_DIR/ml-1m.zip"

echo "📥 Downloading MovieLens 1M..."
mkdir -p "$DATA_DIR"
curl -L "$URL" -o "$ZIP_PATH"

echo "📦 Extracting..."
unzip -o "$ZIP_PATH" -d "$DATA_DIR"
mv "$DATA_DIR/ml-1m/"* "$DATA_DIR/"
rmdir "$DATA_DIR/ml-1m"
rm "$ZIP_PATH"

echo "✅ Done! Files in $DATA_DIR:"
ls -lh "$DATA_DIR"