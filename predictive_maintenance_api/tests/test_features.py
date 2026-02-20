from app.features import FEATURE_NAMES, build_features
from app.schema import SensorSample


def test_build_features_dimension() -> None:
    window = [
        SensorSample(timestamp_sec=i, s1=0.1 + i * 0.01, s2=0.2, s3=0.3, s4=0.4, s5=0.5)
        for i in range(6)
    ]
    feats = build_features(window)
    assert feats.shape[0] == len(FEATURE_NAMES)
