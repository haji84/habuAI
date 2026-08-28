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

# Holdout GPX is stored compressed to keep the repository small. It is expanded
# only in the Actions workspace and therefore remains an unseen validation input
# until the hardened pipeline explicitly scores the holdout period.
if [[ -s data/raw/holdout_2026-08-28.gpx.xz ]]; then
  echo 'Expanding 2026-08-28 holdout GPX...'
  xz -dc data/raw/holdout_2026-08-28.gpx.xz > data/raw/gpx/2026-08-28.gpx
fi

printf 'GPX files: '; find data/raw/gpx -maxdepth 1 -type f -name '*.gpx' | wc -l
printf 'Log files: '; find data/raw/logs -maxdepth 1 -type f -name '*.txt' | wc -l
