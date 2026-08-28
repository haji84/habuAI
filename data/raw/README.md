# Raw data drop zone

Daily operation should require only two uploads:

- `data/raw/gpx/YYYY-MM-DD探索.gpx`
- `data/raw/logs/YYYY-MM-DD-log.txt`

Pushing either file triggers `.github/workflows/habu-pipeline.yml`.

Rules:

- Keep original GPX and log files unchanged.
- Zero-byte GPX files are ignored.
- Timestamps are preserved; GPX UTC timestamps are converted to Asia/Tokyo.
- A survey night is anchored to the explicit `エリア探索開始` record, so post-midnight events remain part of the previous evening's survey until its end.
- Main Habu excludes ヒメハブ, アカマタ, ガラスヒバァ/ガラスヒヴァ, リュウキュウアオヘビ, ヒャン and other snake species.
- Capture-event count and Habu-individual count are separate fields.

Initial model baseline: through 2026-08-27 survey data.
First preferred validation night: 2026-08-28→29.
