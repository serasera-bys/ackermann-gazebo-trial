# Predictive Maintenance API Architecture

## Flow
`windowed sensor samples -> feature engineering -> risk model -> API response`

## Feature blocks
- Rolling means (`s1..s5_mean`)
- Rolling std (`s1..s5_std`)
- Trend slopes (`s1..s5_slope`)
- Delta over window (`delta_s1..delta_s5`)

## Serving contract
- Input: sequence (`>=5`) with `timestamp_sec, s1..s5`
- Output: `risk_score` + `remaining_useful_life_bucket`

## Artifacts
- `model.joblib`
- `feature_schema.json`
- `eval_report.json`
- `train_metadata.json`
