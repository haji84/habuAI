#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/gpx data/raw/logs data/raw/import
shopt -s nullglob
for bundle in data/raw/import/*.zip; do
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

printf 'GPX files: '; find data/raw/gpx -maxdepth 1 -type f -name '*.gpx' | wc -l
printf 'Log files: '; find data/raw/logs -maxdepth 1 -type f -name '*.txt' | wc -l
