#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/gpx data/raw/logs data/raw/import
shopt -s nullglob

bundles=(data/raw/import/*.zip data/raw/*.zip)
for bundle in "${bundles[@]}"; do
  echo "Importing raw bundle: $bundle"
  tmp="$(mktemp -d)"
  unzip -q -o "$bundle" -d "$tmp"
  if [ -d "$tmp/data/raw/gpx" ]; then cp -f "$tmp"/data/raw/gpx/*.gpx data/raw/gpx/ 2>/dev/null || true; fi
  if [ -d "$tmp/data/raw/logs" ]; then cp -f "$tmp"/data/raw/logs/*.txt data/raw/logs/ 2>/dev/null || true; fi
  if [ -d "$tmp/gpx" ]; then cp -f "$tmp"/gpx/*.gpx data/raw/gpx/ 2>/dev/null || true; fi
  if [ -d "$tmp/logs" ]; then cp -f "$tmp"/logs/*.txt data/raw/logs/ 2>/dev/null || true; fi
  rm -rf "$tmp"
done

# The previously uploaded 2026-08-28 compressed reconstruction is known to be
# corrupt. Never allow it to contaminate training and never let it block the
# verified 8/13-8/27 baseline pipeline. The 8/28 night remains an explicit
# holdout until a byte-valid GPX is supplied/rebuilt.
rm -f data/raw/gpx/2026-08-28.gpx
printf 'GPX files: '; find data/raw/gpx -maxdepth 1 -type f -name '*.gpx' -size +0c | wc -l
printf 'Log files: '; find data/raw/logs -maxdepth 1 -type f -name '*.txt' -size +0c | wc -l

echo 'Holdout status: 2026-08-28 GPX quarantined; baseline run continues without leakage.'
