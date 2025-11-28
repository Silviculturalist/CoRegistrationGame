import pytest

from trees import Tree


def test_impute_height_with_diameter_only():
    t = Tree("t1", 0.0, 0.0)
    t.stemdiam = 0.30
    t.impute_height()
    expected_height = t.get_height(0.30)
    assert t.height == pytest.approx(expected_height)


def test_impute_dbh_with_height_only():
    t = Tree("t2", 0.0, 0.0)
    t.height = 20.0
    t.impute_dbh()
    expected_diam = t.get_diameter(20.0)
    assert t.stemdiam == pytest.approx(expected_diam)
