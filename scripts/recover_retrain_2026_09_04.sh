#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$ROOT/staging/retrain-2026-09-04"
OUTDIR="$ROOT/data/raw/gpx"
EXPECTED_SHA="29aeebb78545ec897c363024b813c6d04975c4ed13c7f243cd71d60516134904"
TARGET="$OUTDIR/2026-09-04-探索.gpx"

mkdir -p "$OUTDIR"
TMP_BIN="$(mktemp)"
TMP_DIR="$(mktemp -d)"
trap 'rm -f "$TMP_BIN"; rm -rf "$TMP_DIR"' EXIT

: > "$TMP_BIN"
for part in "$STAGING"/part_*.b64; do
  echo "Decoding $(basename "$part")"
  base64 -d "$part" >> "$TMP_BIN"
done

extract_gpx() {
  local src="$1"
  if grep -aq '<trkpt' "$src"; then
    cp "$src" "$TARGET"
    return 0
  fi

  if file "$src" | grep -qi 'Zip archive'; then
    unzip -q -o "$src" -d "$TMP_DIR"
    local found
    found="$(find "$TMP_DIR" -type f -name '*.gpx' | head -n 1 || true)"
    if [ -n "$found" ]; then
      cp "$found" "$TARGET"
      return 0
    fi
  fi

  if file "$src" | grep -qi 'XZ compressed'; then
    xz -dc "$src" > "$TARGET"
    return 0
  fi

  if file "$src" | grep -qi 'gzip compressed'; then
    gzip -dc "$src" > "$TARGET"
    return 0
  fi

  return 1
}

if ! extract_gpx "$TMP_BIN"; then
  echo 'Unable to recover 2026-09-04 GPX from staging parts.' >&2
  file "$TMP_BIN" >&2 || true
  exit 2
fi

if ! grep -q '<trkpt' "$TARGET"; then
  echo 'Recovered file is not a GPX track.' >&2
  exit 3
fi

ACTUAL_SHA="$(sha256sum "$TARGET" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "SHA mismatch: expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  rm -f "$TARGET"
  exit 4
fi

echo "Recovered canonical GPX: $TARGET"
echo "SHA256=$ACTUAL_SHA"
