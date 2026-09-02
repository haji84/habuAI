# Raw data drop zone

Daily operation should require only two uploads:

- `data/raw/gpx/YYYY-MM-DD探索.gpx`
- `data/raw/logs/YYYY-MM-DD-log.txt`

Pushing either file triggers `.github/workflows/habu-pipeline.yml`.

Rules:

- Keep original GPX and log files unchanged.
- Zero-byte GPX files are ignored.
- Timestamps are preserved; GPX UTC timestamps are converted to Asia/Tokyo before operational-night assignment.
- The single canonical operational-night boundary is 07:00 Asia/Tokyo: 00:00:00 through 06:59:59 belong to the previous operational night; 07:00:00 onward belongs to the calendar date.
- The same `operational_date_0700` rule must be used for GPX, field events, sessions, weather joins, prediction runs, training data and evaluation. Session-start calendar date must not override it.
- Main Habu excludes ヒメハブ, アカマタ, ガラスヒバァ/ガラスヒヴァ, リュウキュウアオヘビ, ヒャン and other snake species.
- Capture-event count and Habu-individual count are separate fields.
- Raw ACTUAL_GPX continuity does not by itself authorize strict Road x 10 min negatives. Road map matching must pass the strict eligibility gate before `NO_CAPTURE_OBSERVED` is generated.

Initial model baseline: through 2026-08-27 survey data.
First preferred validation night: 2026-08-28→29.
