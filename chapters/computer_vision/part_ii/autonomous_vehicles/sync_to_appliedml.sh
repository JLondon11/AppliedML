#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a local clone of JLondon11/AppliedML.
SRC="${1:-.}"
TARGET="chapters/computer_vision/part_ii/autonomous_vehicles"

mkdir -p "$TARGET/figures" "$TARGET/code"

if [ -d "$SRC/chapters/computer_vision/part_ii/autonomous_vehicles" ]; then
  cp -R "$SRC/chapters/computer_vision/part_ii/autonomous_vehicles/"* "$TARGET/"
else
  echo "Expected packaged path not found under: $SRC" >&2
  exit 2
fi

git add "$TARGET"
git commit -m "Add approved Computer Vision Part II figures and reproducibility assets"
git push origin main
