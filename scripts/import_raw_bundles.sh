#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/gpx data/raw/logs data/raw/import
shopt -s nullglob

bundles=(data/raw/import/*.zip data/raw/*.zip)
for bundle in "${bundles[@]}"; do
  echo "Importing raw bundle: $bundle"
  tmp="$(mktemp -d)"
  unzip -q -o "$bundle" -d "$tmp"

  # Accept either a repository-shaped bundle (data/raw/...) or flat gpx/log folders.
  if [ -d "$tmp/data/raw/gpx" ]; then
    cp -f "$tmp"/data/raw/gpx/*.gpx data/raw/gpx/ 2>/dev/null || true
  fi
  if [ -d "$tmp/data/raw/logs" ]; then
    cp -f "$tmp"/data/raw/logs/*.txt data/raw/logs/ 2>/dev/null || true
  fi
  if [ -d "$tmp/gpx" ]; then
    cp -f "$tmp"/gpx/*.gpx data/raw/gpx/ 2>/dev/null || true
  fi
  if [ -d "$tmp/logs" ]; then
    cp -f "$tmp"/logs/*.txt data/raw/logs/ 2>/dev/null || true
  fi

  rm -rf "$tmp"
done

# The holdout archive is stored as base64 text of an xz-compressed GPX because
# GitHub's UTF-8 contents API cannot write arbitrary binary bytes directly.
# Decode to a temporary xz stream, validate it, then expand into the Actions
# workspace. The source remains separate from the frozen baseline bundle.
if [[ -s data/raw/holdout_2026-08-28.gpx.xz ]]; then
  echo 'Decoding and expanding 2026-08-28 holdout GPX...'
  tmp_xz="$(mktemp --suffix=.gpx.xz)"
  if head -c 6 data/raw/holdout_2026-08-28.gpx.xz | grep -q '^/Td6WF'; then
    base64 -d data/raw/holdout_2026-08-28.gpx.xz > "$tmp_xz"
  else
    cp data/raw/holdout_2026-08-28.gpx.xz "$tmp_xz"
  fi
  xz -t "$tmp_xz"
  xz -dc "$tmp_xz" > data/raw/gpx/2026-08-28.gpx
  rm -f "$tmp_xz"
fi

printf 'GPX files: '; find data/raw/gpx -maxdepth 1 -type f -name '*.gpx' | wc -l
printf 'Log files: '; find data/raw/logs -maxdepth 1 -type f -name '*.txt' | wc -l
