import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from habuai.runtime_fixes import _rescue_capture_labels_to_observed_visits


def test_rescues_confirmed_capture_to_observed_visit_within_50m_10min():
    segs = gpd.GeoDataFrame(
        {
            "segment_id": ["A", "B"],
            "geometry": [
                LineString([(129.3900, 28.1645), (129.3901, 28.1645)]),
                LineString([(129.3910, 28.1645), (129.3911, 28.1645)]),
            ],
        },
        crs="EPSG:4326",
    ).to_crs("EPSG:6669")

    visits = pd.DataFrame(
        {
            "segment_id": ["A"],
            "entered_at": [pd.Timestamp("2026-09-05T23:50:00+09:00")],
            "habu_capture": [0],
            "habu_individuals": [0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-09-05T23:54:00+09:00")],
            "event_type": ["捕獲"],
            "species": ["ハブ"],
            "individual_count": [1],
            "segment_id": ["B"],
            "lat": [28.1645],
            "lon": [129.39005],
        }
    )

    out = _rescue_capture_labels_to_observed_visits(visits, events, segs)
    assert int(out.loc[0, "habu_capture"]) == 1
    assert int(out.loc[0, "habu_individuals"]) == 1
    assert out.loc[0, "outcome_label_method"] == "spatiotemporal_fallback_50m_10min"
    assert float(out.loc[0, "label_event_distance_m"]) <= 50.0


def test_does_not_rescue_himehabu():
    segs = gpd.GeoDataFrame(
        {"segment_id": ["A"], "geometry": [LineString([(129.3900, 28.1645), (129.3901, 28.1645)])]},
        crs="EPSG:4326",
    ).to_crs("EPSG:6669")
    visits = pd.DataFrame(
        {
            "segment_id": ["A"],
            "entered_at": [pd.Timestamp("2026-09-05T23:50:00+09:00")],
            "habu_capture": [0],
            "habu_individuals": [0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-09-05T23:54:00+09:00")],
            "event_type": ["捕獲"],
            "species": ["ヒメハブ"],
            "individual_count": [1],
            "segment_id": ["A"],
            "lat": [28.1645],
            "lon": [129.39005],
        }
    )
    out = _rescue_capture_labels_to_observed_visits(visits, events, segs)
    assert int(out.loc[0, "habu_capture"]) == 0
