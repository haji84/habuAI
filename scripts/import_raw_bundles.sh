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

# 2026-08-28 holdout GPX is a compact 5 m-resampled route stored as small,
# individually auditable base64 chunks. The old monolithic compressed file is
# intentionally ignored because it was truncated in an earlier upload.
holdout_parts=(
  data/raw/holdout_parts/part_00_v2.b64
  data/raw/holdout_parts/part_01_v2.b64
  data/raw/holdout_parts/part_02.b64
  data/raw/holdout_parts/part_03.b64
  data/raw/holdout_parts/part_04.b64
  data/raw/holdout_parts/part_05.b64
  data/raw/holdout_parts/part_06a.b64
  data/raw/holdout_parts/part_06c00.b64
  data/raw/holdout_parts/part_06c01.b64
  data/raw/holdout_parts/part_06c02.b64
  data/raw/holdout_parts/part_06c03.b64
  data/raw/holdout_parts/part_06c04.b64
  data/raw/holdout_parts/part_06c05.b64
  data/raw/holdout_parts/part_06c06.b64
  data/raw/holdout_parts/part_06c07.b64
  data/raw/holdout_parts/part_06c08.b64
)

missing=0
for part in "${holdout_parts[@]}"; do
  if [[ ! -s "$part" ]]; then
    echo "Missing holdout chunk: $part" >&2
    missing=1
  fi
done

if [[ "$missing" -eq 0 ]]; then
  echo 'Reassembling and validating 2026-08-28 holdout GPX...'
  tmp_b64="$(mktemp)"
  tmp_xz="$(mktemp --suffix=.gpx.xz)"
  cat "${holdout_parts[@]}" > "$tmp_b64"

  actual_chars="$(wc -c < "$tmp_b64" | tr -d ' ')"
  expected_chars=68988
  if [[ "$actual_chars" -ne "$expected_chars" ]]; then
    echo "Holdout base64 length mismatch: expected=$expected_chars actual=$actual_chars" >&2
    exit 3
  fi

  base64 -d "$tmp_b64" > "$tmp_xz"
  xz -t "$tmp_xz"
  xz -dc "$tmp_xz" > data/raw/gpx/2026-08-28.gpx

  point_count="$(grep -o '<trkpt ' data/raw/gpx/2026-08-28.gpx | wc -l | tr -d ' ')"
  expected_points=9358
  if [[ "$point_count" -ne "$expected_points" ]]; then
    echo "Holdout GPX point-count mismatch: expected=$expected_points actual=$point_count" >&2
    exit 4
  fi

  echo "Holdout GPX validated: $point_count points"
  rm -f "$tmp_b64" "$tmp_xz"
else
  echo 'Holdout chunks are incomplete; refusing to run with a partial validation route.' >&2
  exit 3
fi

printf 'GPX files: '; find data/raw/gpx -maxdepth 1 -type f -name '*.gpx' | wc -l
printf 'Log files: '; find data/raw/logs -maxdepth 1 -type f -name '*.txt' | wc -l
