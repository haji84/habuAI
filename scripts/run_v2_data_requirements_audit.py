from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CAP_DIR=ROOT/'data'/'canonical'
REPORTS=ROOT/'reports'
PROCESSED=ROOT/'data'/'processed'


def nonempty(s: pd.Series)->pd.Series:
    return s.notna() & s.astype(str).str.strip().ne('') & ~s.astype(str).str.lower().isin({'nan','none','null','nat'})

parts=sorted(CAP_DIR.glob('habu_capture_master_part_*.csv'))
cap=pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in parts], ignore_index=True)
aux=pd.read_csv(CAP_DIR/'habu_auxiliary_events.csv', dtype=str, keep_default_na=False)

n=len(cap)
col_stats={}
for c in cap.columns:
    present=int(nonempty(cap[c]).sum())
    col_stats[c]={'present':present,'missing':n-present,'pct':round(present/n*100,2) if n else 0.0}

# Strong core fields for v2
lat='緯度'; lon='経度'; date='日付'; time='時刻'
gps_mask=nonempty(cap[lat]) & nonempty(cap[lon]) if lat in cap and lon in cap else pd.Series(False,index=cap.index)
time_mask=nonempty(cap[time]) if time in cap else pd.Series(False,index=cap.index)
date_mask=nonempty(cap[date]) if date in cap else pd.Series(False,index=cap.index)
size_mask=nonempty(cap['サイズ']) if 'サイズ' in cap else pd.Series(False,index=cap.index)
weather_mask=nonempty(cap['天候']) if '天候' in cap else pd.Series(False,index=cap.index)
temp_mask=nonempty(cap['気温']) if '気温' in cap else pd.Series(False,index=cap.index)
moon_mask=nonempty(cap['月齢']) if '月齢' in cap else pd.Series(False,index=cap.index)
tide_mask=nonempty(cap['潮']) if '潮' in cap else pd.Series(False,index=cap.index)
region_mask=nonempty(cap['地域']) if '地域' in cap else pd.Series(False,index=cap.index)
source_mask=nonempty(cap['ソース']) if 'ソース' in cap else pd.Series(False,index=cap.index)

# Existing validated summaries
canon=json.loads((REPORTS/'canonical_master_audit.json').read_text(encoding='utf-8'))
hurdle=json.loads((REPORTS/'hurdle_count_fast_summary.json').read_text(encoding='utf-8'))
zone=json.loads((REPORTS/'zone_models_89night_pdca_summary.json').read_text(encoding='utf-8'))
five=json.loads((REPORTS/'five_models_89night_pdca_summary.json').read_text(encoding='utf-8'))
hist=json.loads((REPORTS/'historical_route_reconstruction.json').read_text(encoding='utf-8'))

# GPX/exposure inventory, robust to column changes
exposure={}
for fname in ['segment_visits.csv','gpx_points_matched.csv']:
    p=PROCESSED/fname
    if p.exists():
        df=pd.read_csv(p, low_memory=False)
        exposure[fname]={'rows':int(len(df)), 'columns':list(df.columns)}
        night_cols=[c for c in df.columns if c.lower() in {'night','night_date','operational_day','session_date'} or 'night' in c.lower() or '運用日' in c]
        session_cols=[c for c in df.columns if 'session' in c.lower() or 'file' in c.lower()]
        for c in night_cols[:2]:
            exposure[fname][f'unique_{c}']=int(df[c].dropna().astype(str).nunique())
        for c in session_cols[:2]:
            exposure[fname][f'unique_{c}']=int(df[c].dropna().astype(str).nunique())

# requirements audit
requirements=[
    {'domain':'capture_positive','field':'capture event identity/count','status':'ready','evidence':f"{canon['capture_events']} capture events / {canon['capture_individuals']} individuals",'priority':'P0'},
    {'domain':'capture_positive','field':'capture GPS','status':'partial','evidence':f"{canon['gps_capture_events']}/{canon['capture_events']} events ({canon['gps_capture_events']/canon['capture_events']*100:.1f}%)",'priority':'P0'},
    {'domain':'capture_positive','field':'capture time','status':'partial','evidence':f"{int(time_mask.sum())}/{n} canonical rows non-empty; timestamp+GPS validated events={zone['gps_timestamp_capture_events_total']}",'priority':'P0'},
    {'domain':'nightly_activity','field':'verified operating nights','status':'partial','evidence':f"89 field-evidence nights, but these are not proven to be all operating nights",'priority':'P0'},
    {'domain':'nightly_activity','field':'true zero-capture nights','status':'critical_gap','evidence':f"only {hurdle['zero_field_evidence_nights']} weak zero field-evidence nights; not all GPX-confirmed full surveys",'priority':'P0'},
    {'domain':'detection_exposure','field':'actual GPX route/time exposure','status':'critical_gap','evidence':f"strict road-time scoring available on {zone['model_availability']['zone1000']['strict_scored_nights']}/89 field-evidence nights; historical routes are reconstructed for spatial use only",'priority':'P0'},
    {'domain':'detection_exposure','field':'visited-but-no-capture road×time negatives','status':'critical_gap','evidence':'available only where actual timed GPX/visit evidence exists; cannot safely fabricate for older nights','priority':'P0'},
    {'domain':'spatial','field':'historical route reconstruction','status':'partial','evidence':f"{hist['nights_reconstructed']}/{hist['nights_total']} May-Jul capture nights reconstructed; {hist['nights_coverage_100pct']} at 100% capture-segment coverage",'priority':'P1'},
    {'domain':'environment','field':'historical weather','status':'ready_with_proxy','evidence':'Open-Meteo historical cache exists; forecast-vs-observation distinction still required for production-equivalent backtests','priority':'P1'},
    {'domain':'environment','field':'local micro-weather on route','status':'gap','evidence':'canonical weather/temp are incomplete and gridded weather cannot fully represent road-level fog/wetness/microclimate','priority':'P1'},
    {'domain':'road_surface','field':'road wetness / dry-wet-soaked state','status':'critical_gap','evidence':'not present in canonical capture master schema as a structured per-visit road×time field','priority':'P0'},
    {'domain':'observation','field':'search effort duration/distance/start/end','status':'critical_gap','evidence':'not complete across 89 nights; needed to compare capture probability per unit effort','priority':'P0'},
    {'domain':'observation','field':'search speed / repeated passes / direction','status':'gap','evidence':'derivable on GPX nights, unavailable for most historical nights','priority':'P1'},
    {'domain':'biology','field':'sex','status':'critical_gap_for_sex_models','evidence':'not present as a structured canonical capture-master field; cannot infer female from size','priority':'P2'},
    {'domain':'biology','field':'body size','status':'partial','evidence':f"{int(size_mask.sum())}/{n} canonical capture rows ({size_mask.mean()*100:.1f}%)",'priority':'P2'},
    {'domain':'biology','field':'behavior/found position/prey context','status':'partial_or_external','evidence':'some observations exist in broader logs/biological-reaction data, but not normalized into the canonical capture master used by v2 ranking','priority':'P2'},
    {'domain':'context','field':'moon/tide','status':'available_as_derived','evidence':'can be derived deterministically/proxy from timestamp; direct canonical fields are not required if timestamp is valid','priority':'P2'},
    {'domain':'offline_app','field':'offline regional map packs','status':'not_implemented','evidence':'v2 requirement exists; current repository has road/map data for present study area, not Amami+Okinawa regional downloadable packs','priority':'P0_app'},
    {'domain':'offline_app','field':'offline model/runtime/cache synchronization','status':'not_implemented','evidence':'required by v2 but not represented as production mobile data package yet','priority':'P0_app'},
]

# classify model-vs-data bottleneck from existing fair walkforward
best100=None
for row in five.get('overall',[]):
    if row.get('n_predictions')==100:
        v=row.get('coverage',{}).get('100m_20min')
        if v is not None and (best100 is None or v>best100['coverage']):
            best100={'model':row['model'],'coverage':v}

conclusion={
    'data_bottleneck_dominant': True,
    'reasoning': [
        f"Only {canon['gps_capture_events']}/{canon['capture_events']} capture events have GPS, so {canon['gps_missing_capture_events']} historical positives cannot teach exact road location.",
        f"Strict road×time evaluation is possible on only {zone['model_availability']['zone1000']['strict_scored_nights']}/89 field-evidence nights, leaving most nights without trustworthy timed exposure/negative evidence.",
        f"True zero nights are especially weak: {hurdle['zero_field_evidence_nights']} field-evidence zeros and the hurdle report explicitly says they are not all GPX-confirmed full surveys.",
        "Without complete search exposure, the model sees many captures but cannot reliably distinguish 'bad road/time' from 'not searched'. This is a structural identifiability problem, not something a more complex model alone can solve.",
        f"Even after testing five advanced model families on a common candidate space, the best Top100 100m±20min coverage was {best100['coverage']*100:.1f}% ({best100['model']}) if present in the report, supporting the view that model complexity alone is not closing the gap to 90%." if best100 else "Advanced model tests did not establish a path to 90% coverage."
    ],
    'cannot_claim_90pct_possible_from_current_data': True,
}

collection_plan=[
    {'rank':1,'item':'Every search night: GPX from start to finish, including zero-capture nights','why':'creates trustworthy exposure and negatives','target':'100% of future nights'},
    {'rank':2,'item':'Search session start/end, distance, duration, and explicit 0-capture outcome','why':'makes count/detection models identifiable','target':'100% of future nights'},
    {'rank':3,'item':'Every capture/sighting: exact GPS + timestamp','why':'required for 100m±20min target','target':'>=98% future events'},
    {'rank':4,'item':'Road condition at observation/pass: dry/damp/wet/soaked + rainfall/fog state','why':'road-level micro-environment can vary within gridded weather','target':'structured automatic/manual capture for every session/observation'},
    {'rank':5,'item':'Pre-night forecast snapshot with issue time','why':'allows production-equivalent leakage-free weather validation','target':'every prediction night'},
    {'rank':6,'item':'Device/offline state and map-pack version','why':'reproducible offline prediction and route generation','target':'every prediction run'},
    {'rank':7,'item':'Sex/size/behavior when practical','why':'valuable for ecology and large-female impact models but secondary to core route-time accuracy','target':'as complete as safely practical'},
]

out={
    'status':'ok',
    'audit_name':'Habu AI v2 data requirements audit',
    'canonical_capture_rows':n,
    'canonical_columns':list(cap.columns),
    'canonical_column_completeness':col_stats,
    'core_counts':canon,
    'field_evidence_nights':hurdle['field_evidence_nights'],
    'capture_positive_nights':hurdle['capture_positive_nights'],
    'zero_field_evidence_nights':hurdle['zero_field_evidence_nights'],
    'strict_road_time_scored_nights':zone['model_availability']['zone1000']['strict_scored_nights'],
    'historical_route_reconstruction':{k:hist[k] for k in ['nights_total','capture_events','road_matched_capture_events','nights_reconstructed','nights_coverage_100pct','nights_coverage_ge_50pct','confidence_counts']},
    'processed_exposure_inventory':exposure,
    'requirements':requirements,
    'bottleneck_conclusion':conclusion,
    'collection_priority':collection_plan,
    'guardrails':[
        'No missing record is treated as a negative observation.',
        'Reconstructed historical routes may support spatial learning but not fabricated passage times.',
        'Sex is never inferred from body size.',
        'Production-equivalent weather evaluation must use only information available at forecast issue time.',
        'Private road names/coordinates/hotspot rankings are not emitted in this aggregate audit.'
    ]
}
(REPORTS/'v2_data_requirements_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'ok','capture_rows':n,'strict_road_time_scored_nights':out['strict_road_time_scored_nights'],'zero_field_evidence_nights':out['zero_field_evidence_nights']},ensure_ascii=False))
