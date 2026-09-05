from __future__ import annotations

import json
from pathlib import Path


def write_leaflet_route_map(
    plan_payload: dict[str, object],
    route_geojson: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Write a mobile-friendly Leaflet map using exact GeoJSON geometry."""
    plan_json = json.dumps(plan_payload, ensure_ascii=False).replace("</", "<\\/")
    geo_json = json.dumps(route_geojson, ensure_ascii=False).replace("</", "<\\/")

    html = f"""<!doctype html>
<html lang=\"ja\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">
<title>ハブAI 今夜の作戦</title>
<link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\">
<style>
html,body,#map{{height:100%;margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
#panel{{position:absolute;z-index:1000;left:10px;right:10px;top:10px;background:rgba(255,255,255,.96);border-radius:14px;padding:12px;box-shadow:0 3px 18px rgba(0,0,0,.22);max-height:42vh;overflow:auto}}
.title{{font-weight:800;font-size:18px;margin-bottom:6px}}
.meta{{font-size:13px;line-height:1.45;margin-bottom:8px}}
.route{{border-top:1px solid #ddd;padding:8px 0;font-size:13px}}
.route strong{{font-size:14px}}
button{{border:0;border-radius:10px;padding:8px 10px;margin:4px 4px 0 0;font-weight:700}}
</style>
</head>
<body>
<div id=\"map\"></div>
<div id=\"panel\"><div id=\"content\"></div></div>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
<script>
const plan={plan_json};
const routes={geo_json};
const colors={{A_CAPTURE_MAX:'#d7191c',B_EFFICIENCY:'#2c7bb6',C_ALTERNATIVE:'#7b3294'}};
const map=L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:20,attribution:'&copy; OpenStreetMap contributors'}}).addTo(map);
const layers={{}};
for(const kind of ['A_CAPTURE_MAX','B_EFFICIENCY','C_ALTERNATIVE']){{
  const featureSet={{type:'FeatureCollection',features:routes.features.filter(f=>f.properties && f.properties.route_kind===kind && f.geometry)}};
  layers[kind]=L.geoJSON(featureSet,{{style:()=>({{color:colors[kind],weight:6,opacity:.85}}),onEachFeature:(f,l)=>l.bindPopup(`${{f.properties.route_label}}<br>順番 ${{f.properties.route_order}}<br>segment: ${{f.properties.segment_id}}`)}}).addTo(map);
}}
const all=L.featureGroup(Object.values(layers));
if(all.getBounds().isValid()) map.fitBounds(all.getBounds(),{{padding:[30,30]}}); else map.setView([28.17,129.32],12);
const content=document.getElementById('content');
content.innerHTML=`<div class=\"title\">今夜の作戦 ${{plan.exploration_night}}</div><div class=\"meta\">一点予測: <b>${{plan.point_prediction}}匹</b><br>本命: ${{plan.primary_window}}${{plan.secondary_window?'<br>対抗: '+plan.secondary_window:''}}</div>`;
for(const r of plan.routes){{
  const div=document.createElement('div'); div.className='route';
  div.innerHTML=`<strong>${{r.label}}</strong><br>${{r.explanation}}<br>期待 ${{Number(r.expected_captures).toFixed(2)}} / ${{Number(r.distance_km).toFixed(1)}}km / ${{Math.round(r.duration_min)}}分<br><button data-kind=\"${{r.kind}}\">このルートだけ表示</button>`;
  content.appendChild(div);
}}
const reset=document.createElement('button'); reset.textContent='3ルート表示'; reset.onclick=()=>{{Object.values(layers).forEach(l=>map.addLayer(l)); if(all.getBounds().isValid()) map.fitBounds(all.getBounds(),{{padding:[30,30]}})}}; content.appendChild(reset);
content.addEventListener('click',e=>{{if(!e.target.dataset.kind)return; const kind=e.target.dataset.kind; Object.entries(layers).forEach(([k,l])=>{{if(k===kind)map.addLayer(l);else map.removeLayer(l)}}); const b=layers[kind].getBounds(); if(b.isValid())map.fitBounds(b,{{padding:[30,30]}})}});
</script>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
