import pandas as pd

from habuai.hardening.spatial_ablation import _group_columns,_full_features


def test_spatial_ablation_groups_are_disjoint_and_cover_history_families():
    cols={
        "length_m":1.0,"bearing_deg":0.0,"curvature_deg":0.0,"road_class_code":1.0,"junction_distance_m":5.0,
        "stream_distance_m":10.0,"coast_distance_m":20.0,"forest_distance_m":30.0,"farmland_distance_m":40.0,"residential_distance_m":50.0,
        "hist_capture_count_50m":1,"hist_capture_30d_count_50m":1,"hist_capture_90d_count_50m":1,"days_since_capture_50m":2.0,
        "hist_large_capture_count_50m":0,"hist_bio_count_50m":3,"hist_bio_30d_count_50m":2,
        "hist_bio_カエル_30d_count_50m":1,
    }
    data=pd.DataFrame([cols])
    groups=_group_columns(data)
    assert set(groups)=={"road_geometry","terrain_context","capture_alltime","capture_30d","capture_90d","capture_recency","large_capture_history","bio_alltime","bio_30d_total","bio_species_30d"}
    flattened=[c for values in groups.values() for c in values]
    assert len(flattened)==len(set(flattened))
    full=_full_features(data)
    assert "days_since_capture_50m" in full
    assert "hist_bio_カエル_30d_count_50m" in full
