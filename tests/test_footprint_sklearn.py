"""
Unit tests for Scikit-Learn model footprint analyzer.
"""

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

from lodevem.footprint.analyzers.sklearn import analyze_sklearn_footprint


def test_decision_tree_footprint(tmp_path):
    model = DecisionTreeRegressor(max_depth=3)
    X = np.random.randn(20, 2)
    y = np.random.randn(20)
    model.fit(X, y)

    model_file = tmp_path / "dt_model.joblib"
    joblib.dump(model, model_file)

    from lodevem.security import SecurityPolicyError

    with pytest.raises(SecurityPolicyError):
        analyze_sklearn_footprint(model_file)

    fp = analyze_sklearn_footprint(model_file, allow_untrusted=True)

    assert fp.backend == "sklearn"
    assert fp.estimator_type == "DecisionTreeRegressor"
    assert fp.max_depth == 3
    assert fp.total_node_count is not None
    assert fp.total_node_count > 0
    assert fp.n_features_in == 2
    # Ensure honest metric reporting: no fake NN parameters or FLOPs
    assert fp.total_parameters is None
    assert fp.parameter_memory_mb is None
    assert fp.flops is None
    assert fp.flops_status == "not_applicable"
    assert fp.metrics_status["flops"] == "not_applicable"
    assert fp.metrics_status["parameters"] == "not_applicable"


def test_random_forest_footprint(tmp_path):
    rf = RandomForestClassifier(n_estimators=4, max_depth=2, random_state=42)
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, size=30)
    rf.fit(X, y)

    model_file = tmp_path / "rf_model.pkl"
    joblib.dump(rf, model_file)

    from lodevem.security import SecurityPolicyError

    with pytest.raises(SecurityPolicyError):
        analyze_sklearn_footprint(model_file)

    fp = analyze_sklearn_footprint(model_file, allow_untrusted=True)

    assert fp.backend == "sklearn"
    assert fp.estimator_type == "RandomForestClassifier"
    assert fp.n_estimators == 4
    assert fp.max_depth == 2
    assert fp.n_features_in == 5
    assert fp.total_node_count is not None
    assert fp.total_node_count >= 4  # At least 1 root + children per tree
    assert fp.flops_status == "not_applicable"
