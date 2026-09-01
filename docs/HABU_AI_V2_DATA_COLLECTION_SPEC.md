# Habu AI v2 データ収集仕様書 v1.0

更新日: 2026-09-02

## 1. 目的

Habu AI v2 は、夜間探索開始前に「どこを、いつ、どの順番で探索するか」を予測し、限られた探索時間・走行距離の中で捕獲期待値を最大化する。

データ設計では、捕獲DBから探索夜を逆算しない。探索セッション・GPX・GPS履歴などの探索証拠を先に確定し、その後に捕獲・目撃・轢死・その他生物反応を時系列で結合する。

## 2. 探索夜の正本定義

### 2.1 運用日

- 07:00〜23:59 のイベントは暦日を探索夜とする。
- 00:00〜06:59 のイベントは前日を探索夜とする。
- 5:00区切りなど旧ルールは使用しない。
- 全イベント、GPX、GPS履歴、予測、探索セッションに同一の `operational_date_0700` を持たせる。

### 2.2 探索夜の母集団

探索夜は以下の探索証拠から作成する。

1. 完全な時刻付き実GPX
2. 端末GPS履歴・位置履歴
3. 探索開始・終了記録
4. 捕獲・目撃・生物反応イベントとそのGPS/時刻
5. その他、実際に探索したことを確認できる記録

捕獲記録が存在すること自体を探索夜の必要条件にしてはならない。

## 3. イベントラベル

### 3.1 Positive capture/event

実際の捕獲、目撃、その他の生物反応はイベント種別を保持したまま保存する。捕獲・目撃・轢死を同一ラベルへ潰さない。

### 3.2 NO_CAPTURE_OBSERVED

探索証拠により道路・時刻を実際に探索または通過したことが確認でき、その道路・時間帯に捕獲イベントが存在しない場合は `NO_CAPTURE_OBSERVED` とする。

これは「ハブが存在しなかった」という生物学的証明ではなく、「その探索条件では発見・捕獲されなかった観測」である。

### 3.3 Unknown

探索した証拠がない道路・時間、または復元精度が不十分な場合は `Unknown` とする。未探索を負例へ変換してはならない。

## 4. 89夜監査の修正

旧監査の `82捕獲夜 + 7ゼロ候補夜` と、後続監査の `87捕獲夜 + 2明示ゼロ夜` は、同じ母集団を同じ方法で集計した数字ではない。

原因は以下。

- 旧Hurdle母集団は「何らかのイベント記録がある夜」から作成され、目撃・轢死だけの夜も0匹候補へ入っていた。
- 2025-11-07 は `capture_or_sighting / 捕獲候補` で確定捕獲扱いが版によって異なった。
- 後から復旧された捕獲夜が存在する。
- 新監査は明示 `no_capture` だけを0夜として扱い、探索証拠起点ではなかった。

したがって、今後 `87+2=89` を最終正解として固定しない。89夜は探索証拠から再構築し、各夜へ捕獲イベントをJOINして分類する。

## 5. 89夜の必須5分類

全89夜を必ず以下のどれか1つへ分類する。

1. `ACTUAL_GPX` / 完全実GPX
2. `RECONSTRUCTED_GPS_HIGH` / GPS履歴から高精度復元可能
3. `RECONSTRUCTED_PARTIAL` / 部分復元可能
4. `SPATIAL_ONLY_RECONSTRUCTION` / 空間のみ復元可能
5. `UNRECONSTRUCTABLE` / 復元不能

各夜について最低限以下を記録する。

- `operational_date_0700`
- classification
- evidence summary
- source files / source logs
- GPS anchor count
- actual GPX point count
- route coverage
- temporal coverage
- `route_confidence`
- `time_confidence`
- 1km Area学習可否
- Road学習可否
- Road×10min学習可否
- Road×10min評価可否
- NO_CAPTURE_OBSERVED生成可否
- limitation reason

## 6. trajectory provenance

実GPXと復元軌跡を混同しない。

- `ACTUAL_GPX`: 実測連続時刻付きGPX
- `RECONSTRUCTED_GPS`: 端末GPS履歴等から復元
- `SPATIAL_ONLY_RECONSTRUCTION`: 経路・位置は復元できるが厳密通過時刻は不明

復元した経路を実測GPXへ上書きしてはならない。

## 7. confidence

`route_confidence` と `time_confidence` を分離して保存する。

初期運用閾値:

- Road×10min 厳密評価候補: `route_confidence >= 0.85` かつ `time_confidence >= 0.80`
- Road空間学習候補: `route_confidence >= 0.65`
- 1km Area学習候補: `route_confidence >= 0.40` または信頼できる空間アンカーあり
- 閾値未満はUnknownとして時間負例を生成しない

この閾値は監査後に感度分析し、固定値として再承認する。

## 8. GPS復元ルール

- 緯度・経度・時刻が十分密なら時系列順にGPX相当軌跡を再構成する。
- 捕獲・目撃イベントのGPS/時刻をアンカーとして使う。
- 道路ネットワークへマップマッチングする。
- 点間の経路候補が複数ある場合、最短路だけを事実として採用しない。
- 現実的な移動速度・道路接続・中間GPS点で候補を絞る。
- 根拠のない通過時刻を生成しない。
- 部分的に高信頼な区間だけをRoad×10minへ使用できる。

## 9. exploration_sessions

探索開始時に `session_id` を自動生成する。

主要項目:

- session_id
- operational_date_0700
- start_at / end_at
- start_lat / start_lon / end_lat / end_lon
- planned_search_minutes
- max_driving_km
- actual_search_minutes
- actual_driving_km
- regional_pack_id / version
- offline_state
- abnormal_end_flag
- catch_count
- sighting_count
- biological_event_count
- trajectory_source
- route_confidence
- time_confidence
- data_quality_class

捕獲0は手入力必須にしない。探索セッションが成立し、捕獲イベントが0なら夜単位の非捕獲結果を導出する。

## 10. gpx_points

探索開始から終了まで自動記録する。

必須:

- session_id
- recorded_at
- latitude
- longitude
- horizontal_accuracy_m
- speed_mps
- heading_deg
- source

GPS停止・精度劣化・長時間ギャップもイベントとして記録する。

## 11. biological_events

主要項目:

- event_id
- session_id
- operational_date_0700
- occurred_at
- event_type
- species
- count
- size_cm / size_class
- sex
- latitude / longitude
- gps_accuracy_m
- location_source
- nearest_road_id
- road_position
- behavior
- road_surface
- weather_observation
- memo

ハブ捕獲DBは本人が実際に捕獲したハブだけを格納し、目撃・他者捕獲・轢死は別イベントとして保持する。

## 12. road_condition_events / weather_observation_events

一晩1値ではなく変更イベント方式を採用する。

路面:
- dry
- damp
- wet
- soaked

雨・霧も開始/終了/強度変更をイベントとして記録する。

## 13. forecast_snapshots

予測生成時点の予報を凍結保存する。後から実測値で置換しない。

主要項目:

- forecast_snapshot_id
- forecast_issued_at
- forecast_acquired_at
- prediction_generated_at
- forecast_age_minutes
- temperature
- precipitation_probability
- precipitation_amount
- humidity
- wind
- cloud_cover
- pressure
- weather_code

オフライン時は最後にキャッシュした予報を使用し、その古さを保存する。

## 14. prediction_runs

初回予測と捕獲後の再予測を上書きしない。

- prediction_run_id
- session_id
- generated_at
- trigger_type
- current_position
- remaining_minutes
- remaining_distance_km
- forecast_snapshot_id
- model_version
- ranked_areas
- ranked_road_zones
- route_plan

## 15. data_quality_audits

探索終了時に自動監査する。

- GPX開始・終了
- GPXギャップ
- GPS精度
- イベントGPS率
- forecast snapshot有無
- road condition有無
- pack version
- trajectory provenance
- reconstruction quality
- route_confidence
- time_confidence

品質例:
- Gold
- Silver
- Bronze
- Spatial-only
- Unusable-for-temporal-negative

## 16. オフライン要件

以下は完全オフラインで動作可能にする。

- GPS
- GPX記録
- オフライン地図
- 捕獲/目撃/生物反応登録
- 予測
- Road ranking
- route generation / reroute
- 捕獲後の再予測

地域パック削除時も捕獲DB、GPX、ユーザーデータ、学習データを削除しない。

## 17. 評価

正式KPI:

- 捕獲/event within 100m + ±20min

補助:

- 50m ±10min
- 100m ±10min
- 100m ±30min
- 250m ±30min

Stage 1:

- Recall >= 90% in Top10 1km Areas

100m±20分で90%は長期研究目標であり、実測で達成するまで性能値として表示しない。

## 18. 次工程

この仕様を正本とし、次の順序で進める。

1. 89夜を探索証拠起点で再監査
2. 高精度復元候補を道路ネットワークへマップマッチング
3. DBスキーマ確定
4. regional pack仕様
5. training dataset builder
6. v2モデル開発

モデル開発を先行させず、まず探索夜・軌跡・負例の意味を固定する。
