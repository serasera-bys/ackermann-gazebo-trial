from hybrid_nav_semantic_rl.scoring_utils import FEATURE_ORDER, linear_policy_score, rule_score


def test_rule_score_ordering() -> None:
    good = {
        "distance_to_frontier": 1.0,
        "estimated_free_gain": 2.0,
        "heading_change": 0.2,
        "local_obstacle_risk": 0.1,
        "semantic_novelty_score": 0.8,
        "semantic_priority_score": 0.7,
    }
    bad = {
        "distance_to_frontier": 3.0,
        "estimated_free_gain": 0.2,
        "heading_change": 1.0,
        "local_obstacle_risk": 0.9,
        "semantic_novelty_score": 0.1,
        "semantic_priority_score": 0.0,
    }
    assert rule_score(good, 1.0, 1.0, 1.0, 1.0, 1.0) > rule_score(bad, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_linear_policy_score() -> None:
    features = {k: 1.0 for k in FEATURE_ORDER}
    weights = {k: 0.5 for k in FEATURE_ORDER}
    score = linear_policy_score(features, weights, bias=1.0)
    assert abs(score - (1.0 + 0.5 * len(FEATURE_ORDER))) < 1e-6
