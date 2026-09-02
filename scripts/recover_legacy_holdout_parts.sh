#!/usr/bin/env bash
set -euo pipefail

PARTS_DIR="data/raw/holdout_parts"
OUT_DIR="data/raw/gpx"
TARGET_SHA="d10ef5ec68db9bef4e792ecb0a8cee418e9277872f09d536363b22b49bce7ee5"
OUTPUT="$OUT_DIR/2026-08-28-recovered-legacy-parts.gpx"

mkdir -p "$OUT_DIR"

if [ ! -d "$PARTS_DIR" ]; then
  echo "No legacy holdout parts directory; nothing to recover."
  exit 0
fi

# The tree contains two historical variants for part_00 and part_01. Other
# part_06* files are subdivisions of part_06, not additional payload. Try the
# four meaningful combinations and accept a result ONLY when the decompressed
# raw GPX SHA-256 equals the independently verified canonical 2026-08-28 raw.
variants=(
  "part_00.b64 part_01.b64"
  "part_00_v2.b64 part_01.b64"
  "part_00.b64 part_01_v2.b64"
  "part_00_v2.b64 part_01_v2.b64"
)

tail_parts=(part_02.b64 part_03.b64 part_04.b64 part_05.b64 part_06.b64)

for variant in "${variants[@]}"; do
  read -r p0 p1 <<< "$variant"
  required=("$p0" "$p1" "${tail_parts[@]}")
  missing=0
  for part in "${required[@]}"; do
    if [ ! -s "$PARTS_DIR/$part" ]; then
      echo "Legacy candidate skipped; missing $PARTS_DIR/$part"
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || continue

  tmp_b64="$(mktemp)"
  tmp_xz="$(mktemp)"
  tmp_gpx="$(mktemp)"
  trap 'rm -f "$tmp_b64" "$tmp_xz" "$tmp_gpx"' RETURN

  cat "$PARTS_DIR/$p0" "$PARTS_DIR/$p1" \
      "$PARTS_DIR/part_02.b64" "$PARTS_DIR/part_03.b64" \
      "$PARTS_DIR/part_04.b64" "$PARTS_DIR/part_05.b64" \
      "$PARTS_DIR/part_06.b64" > "$tmp_b64"

  if ! base64 -d "$tmp_b64" > "$tmp_xz" 2>/dev/null; then
    echo "Legacy candidate failed base64 decode: $p0 + $p1"
    rm -f "$tmp_b64" "$tmp_xz" "$tmp_gpx"
    trap - RETURN
    continue
  fi
  if ! xz -t "$tmp_xz" 2>/dev/null; then
    echo "Legacy candidate failed xz integrity: $p0 + $p1"
    rm -f "$tmp_b64" "$tmp_xz" "$tmp_gpx"
    trap - RETURN
    continue
  fi
  if ! xz -dc "$tmp_xz" > "$tmp_gpx"; then
    echo "Legacy candidate failed decompression: $p0 + $p1"
    rm -f "$tmp_b64" "$tmp_xz" "$tmp_gpx"
    trap - RETURN
    continue
  fi
  if [ ! -s "$tmp_gpx" ] || ! grep -q '<trkpt' "$tmp_gpx"; then
    echo "Legacy candidate is not a usable GPX: $p0 + $p1"
    rm -f "$tmp_b64" "$tmp_xz" "$tmp_gpx"
    trap - RETURN
    continue
  fi

  sha="$(sha256sum "$tmp_gpx" | awk '{print $1}')"
  echo "Legacy candidate $p0 + $p1 -> raw_sha=$sha"
  if [ "$sha" = "$TARGET_SHA" ]; then
    mv -f "$tmp_gpx" "$OUTPUT"
    rm -f "$tmp_b64" "$tmp_xz"
    trap - RETURN
    echo "Recovered canonical 2026-08-28 GPX from legacy parts: $OUTPUT"
    exit 0
  fi

  rm -f "$tmp_b64" "$tmp_xz" "$tmp_gpx"
  trap - RETURN
done

echo "Legacy holdout parts did not reproduce canonical 2026-08-28 raw SHA; keeping them quarantined."
exit 0
