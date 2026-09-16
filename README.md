# habuAI

Automated Habu field-data pipeline for Setouchi, Amami Oshima.

Core workflow: GPX/log ingestion → OSM 10 m road segmentation → map matching → exposure/zero data → weather/biological/terrain features → model evaluation → nightly prediction → Route Optimizer → Top-3 Route Generator → GIS-safe Leaflet map.

## Nightly top-3 route map

`Route Optimizer` should emit multiple road-valid route candidates. HabuAI then selects three deliberately different plans:

- A: expected-capture maximum
- B: efficiency maximum
- C: alternative route emphasizing large-habu / novel high-value segments

Field constraints are applied before route selection. For example, `forest_road_allowed=false` removes forest-road routes from all three candidates without changing the biological occurrence model.

The route map generator never invents road geometry. It renders only existing 10 m GIS road-segment GeoJSON.

Example:

```bash
python scripts/generate_route_plan.py \
  --candidates reports/optimizer/2026-09-05_candidates.json \
  --segments data/processed/road_segments_10m.geojson \
  --night 2026-09-05 \
  --point-prediction 2 \
  --primary-window '22:20-23:30' \
  --secondary-window '00:20-01:20' \
  --max-duration-min 210 \
  --max-overlap 0.70
```

Without `--forest-road-allowed`, forest-road candidates are excluded. Outputs are:

- `<night>_route_plan.json`: mobile UI / scoring contract
- `<night>_routes.geojson`: exact GIS route geometry
- `<night>_route_map.html`: mobile-friendly Leaflet comparison map
