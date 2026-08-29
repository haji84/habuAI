#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/cache data/osm
PBF=data/cache/kyushu-latest.osm.pbf
URL=https://download.geofabrik.de/asia/japan/kyushu-latest.osm.pbf

if [[ ! -s "$PBF" ]]; then
  echo "Downloading Kyushu OSM PBF..."
  curl -fL --retry 4 --retry-delay 5 "$URL" -o "$PBF.tmp"
  mv "$PBF.tmp" "$PBF"
fi

# bbox = west,south,east,north. Keep tracks/service roads by filtering only on highway=*.
osmium extract -b 129.265,28.120,129.425,28.235 "$PBF" \
  -o data/osm/setouchi-all.pbf --overwrite
osmium tags-filter data/osm/setouchi-all.pbf w/highway \
  -o data/osm/setouchi-roads.pbf --overwrite
osmium export data/osm/setouchi-roads.pbf \
  -f geojson -o data/osm/setouchi-roads.geojson --overwrite

# Real environmental context for distance features. These are OSM vector features,
# not placeholders: waterways, coastline, forest/wood, farmland and residential landuse.
osmium tags-filter data/osm/setouchi-all.pbf \
  w/waterway \
  w/natural=coastline \
  w/natural=wood \
  w/landuse=forest,orchard,farmland,farmyard,meadow,residential \
  -o data/osm/setouchi-context.pbf --overwrite
osmium export data/osm/setouchi-context.pbf \
  -f geojson -o data/osm/setouchi-context.geojson --overwrite

echo "OSM extracts ready: data/osm/setouchi-roads.geojson + data/osm/setouchi-context.geojson"
