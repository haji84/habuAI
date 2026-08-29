import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from habuai.hardening.events import collapse_nearest_event_matches, dedupe_events, species_from_text, strict_holdout_score
from habuai.hardening.roads import recover_supplemental_roads


def test_himehabu_not_contaminated_as_habu():
    assert species_from_text("ヒメハブ捕獲") == "ヒメハブ"
    assert species_from_text("ハブ捕獲大") == "ハブ"


def test_dedupe_preserves_multi_individual_count():
    row = {
        "timestamp": pd.Timestamp("2026-08-20T01:44:00+09:00"),
        "species": "ハブ",
        "event_type": "捕獲",
        "lat": 28.1,
        "lon": 129.3,
        "individual_count": 2,
        "raw_text": "ハブ捕獲小2匹",
    }
    out, removed = dedupe_events(pd.DataFrame([row, row]))
    assert len(out) == 1
    assert removed == 1
    assert int(out.iloc[0].individual_count) == 2


def test_nearest_join_tie_collapse_keeps_distinct_source_events():
    joined = pd.DataFrame(
        {
            "event_signature": ["same", "same", "same"],
            "segment_id": ["OSM_B", "OSM_A", "OSM_C"],
            "event_match_distance_m": [12.0, 12.0, 8.0],
        },
        index=[0, 0, 1],
    )
    out = collapse_nearest_event_matches(joined)
    assert len(out) == 2
    assert out.event_signature.tolist() == ["same", "same"]
    assert out.segment_id.tolist() == ["OSM_A", "OSM_C"]


def test_gpx_recovery_requires_event_corroboration():
    base = pd.Timestamp("2026-08-16T00:20:00+09:00")
    rows = []
    for i in range(40):
        rows.append({"session_file":"2026-08-15.gpx","seq":i,"timestamp":base+pd.Timedelta(seconds=i),"lat":28.1702,"lon":129.3000+i*0.00001,"segment_id":None})
    points = pd.DataFrame(rows)
    osm = gpd.GeoDataFrame({"segment_id":["OSM_X"],"geometry":[LineString([(129.29,28.16),(129.291,28.16)])]},crs="EPSG:4326").to_crs("EPSG:6669")
    events = pd.DataFrame([{"timestamp":base+pd.Timedelta(seconds=20),"lat":28.1702,"lon":129.3002,"species":"ハブ","event_type":"捕獲","segment_id":None}])
    recovered,audit = recover_supplemental_roads(points,points,events,osm,{"segment_length_m":10.0})
    assert len(recovered) > 0
    assert recovered.segment_id.str.startswith("GPXROAD_").all()
    assert audit and audit[0]["corroborated_events"][0]["species"] == "ハブ"


def test_strict_holdout_windows_do_not_move():
    rows = []
    for ts in ["2026-08-28T22:06:00+09:00", "2026-08-28T22:45:00+09:00", "2026-08-29T01:22:00+09:00", "2026-08-29T02:26:00+09:00"]:
        rows.append({"timestamp": pd.Timestamp(ts), "species": "ハブ", "event_type": "捕獲", "individual_count": 1})
    score = strict_holdout_score(pd.DataFrame(rows))
    assert score["main_window"]["actual_capture_events"] == 0
    assert score["secondary_window"]["actual_capture_events"] == 1
    assert score["actual_individuals"] == 4
    assert score["range_hit"] is True
