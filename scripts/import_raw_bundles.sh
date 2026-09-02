#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/gpx data/raw/logs data/raw/import data/raw/compressed_gpx
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

# Historical 2026-08-28 data was previously uploaded as split base64 parts.
# Try to reconstruct it, but accept it only if the decompressed raw GPX matches
# the independently verified canonical SHA-256. A failed reconstruction remains
# quarantined and never blocks the rest of the pipeline.
bash scripts/recover_legacy_holdout_parts.sh

# Validated compressed GPX originals may be stored in data/raw/compressed_gpx.
# This path is intentionally separate from the legacy root-level holdout file.
# Every .xz is integrity-tested and its decompressed XML is minimally validated
# before it is allowed into data/raw/gpx.
for compressed in data/raw/compressed_gpx/*.gpx.xz; do
  echo "Checking compressed GPX: $compressed"
  if ! xz -t "$compressed"; then
    echo "WARNING: corrupt xz skipped: $compressed" >&2
    continue
  fi

  basename_xz="$(basename "$compressed")"
  output_name="${basename_xz%.xz}"
  tmp_gpx="$(mktemp)"
  if ! xz -dc "$compressed" > "$tmp_gpx"; then
    echo "WARNING: decompression failed: $compressed" >&2
    rm -f "$tmp_gpx"
    continue
  fi

  if [ ! -s "$tmp_gpx" ]; then
    echo "WARNING: empty GPX skipped: $compressed" >&2
    rm -f "$tmp_gpx"
    continue
  fi
  if ! grep -q '<trkpt' "$tmp_gpx"; then
    echo "WARNING: no GPX track points found; skipped: $compressed" >&2
    rm -f "$tmp_gpx"
    continue
  fi

  mv -f "$tmp_gpx" "data/raw/gpx/$output_name"
  echo "Imported validated compressed GPX: data/raw/gpx/$output_name"
done

# IMPORTANT: data/raw/holdout_2026-08-28.gpx.xz is a legacy corrupt
# reconstruction and is deliberately NOT part of the compressed_gpx glob above.
# A canonical legacy-parts reconstruction or a new byte-valid 2026-08-28
# original under data/raw/compressed_gpx is allowed.

printf 'GPX files: '; find data/raw/gpx -maxdepth 1 -type f -name '*.gpx' -size +0c | wc -l
printf 'Log files: '; find data/raw/logs -maxdepth 1 -type f -name '*.txt' -size +0c | wc -l

echo 'Legacy root holdout remains quarantined; only canonical-SHA legacy recovery or validated compressed originals are accepted.'
