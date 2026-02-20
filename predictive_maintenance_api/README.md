# Predictive Maintenance API

Time-series inspired risk scoring service for machine-health prediction.

## What it provides
- Trainable classifier (`xgboost` if available, fallback to `gradient_boosting`).
- Feature schema artifact for serving parity.
- FastAPI endpoints:
  - `POST /predict`
  - `POST /batch_predict`
  - `GET /model/info`
  - `GET /health`

## Setup
```bash
cd predictive_maintenance_api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model and write artifacts
```bash
python3 -m app.train --data data/cmapss_like.csv --output-dir artifacts
```

Generated artifacts:
- `artifacts/model.joblib`
- `artifacts/feature_schema.json`
- `artifacts/eval_report.json`
- `artifacts/train_metadata.json`

## Run API
```bash
uvicorn app.main:app --reload --port 8090
```

Open Swagger UI at `http://127.0.0.1:8090/docs`.
Root `/` is not implemented, so `404` at `/` is expected.

## Example request
```bash
curl -X POST http://127.0.0.1:8090/predict \
  -H 'Content-Type: application/json' \
  -d @data/sample_request.json
```

## Tests
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Notes
- Default training data is synthetic if `data/cmapss_like.csv` does not exist.
- Replace with real C-MAPSS-derived engineered features for production-grade results.
