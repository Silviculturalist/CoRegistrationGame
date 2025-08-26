import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trees import Stand, Tree
from chm_plot import CHMPlot


def build_stand(tmp_path, impute_dbh=True, impute_h=True):
    data = [
        {
            'Stand': 1,
            'PLOT': 1,
            'TreeID': 't1',
            'X_GROUND': 0.0,
            'Y_GROUND': 0.0,
            'STEMDIAM': 30.0,
            'H': 15.0,
        },
        {
            'Stand': 1,
            'PLOT': 1,
            'TreeID': 't2',
            'X_GROUND': 1.0,
            'Y_GROUND': 1.0,
            'STEMDIAM': 'bad',  # missing diameter -> should be derived from height
            'H': 20.0,
        },
        {
            'Stand': 1,
            'PLOT': 1,
            'TreeID': 't3',
            'X_GROUND': 2.0,
            'Y_GROUND': 2.0,
            'STEMDIAM': 30.0,
            'H': 'bad',  # missing height -> should be derived from diameter
        },
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'trees.csv'
    df.to_csv(csv_path, index=False)
    mapping = {
        'StandID': 'Stand',
        'PlotID': 'PLOT',
        'TreeID': 'TreeID',
        'X': 'X_GROUND',
        'Y': 'Y_GROUND',
        'DBH': 'STEMDIAM',
        'H': 'H',
    }
    return Stand(1, csv_path, mapping=mapping, sep=',', impute_dbh=impute_dbh, impute_h=impute_h)


def build_chm_files(tmp_path):
    height_df = pd.DataFrame([
        {
            'IDALS': 'c1',
            'X': 0.0,
            'Y': 0.0,
            'H': 15.0,
        }
    ])
    h_path = tmp_path / 'chm_h.csv'
    height_df.to_csv(h_path, index=False)

    dbh_df = pd.DataFrame([
        {
            'IDALS': 'c2',
            'X': 1.0,
            'Y': 1.0,
            'DBH': 30.0,
        }
    ])
    d_path = tmp_path / 'chm_d.csv'
    dbh_df.to_csv(d_path, index=False)
    return h_path, d_path


def test_height_parsing_and_derivation(tmp_path):
    stand = build_stand(tmp_path)
    plot = stand.plots[0]
    t1, t2, t3 = plot.trees

    # Row with both height and diameter
    assert t1.stemdiam == pytest.approx(0.30)
    assert t1.height == pytest.approx(15.0)

    # Diameter missing -> derived from height
    expected_diam = t2.get_diameter(20.0)
    assert t2.stemdiam == pytest.approx(expected_diam)
    assert t2.height == pytest.approx(20.0)

    # Height missing -> derived from diameter
    expected_height = t3.get_height(0.30)
    assert t3.stemdiam == pytest.approx(0.30)
    assert t3.height == pytest.approx(expected_height)


def test_stand_no_impute(tmp_path):
    stand = build_stand(tmp_path, impute_dbh=False, impute_h=False)
    plot = stand.plots[0]
    t1, t2, t3 = plot.trees

    assert t1.stemdiam == pytest.approx(0.30)
    assert t1.height == pytest.approx(15.0)
    assert t2.stemdiam is None
    assert t2.height == pytest.approx(20.0)
    assert t3.stemdiam == pytest.approx(0.30)
    assert t3.height is None


def test_chm_plot_imputation(tmp_path):
    h_path, d_path = build_chm_files(tmp_path)

    chm = CHMPlot(h_path, sep=',')
    t = chm.trees[0]
    assert t.stemdiam is None
    assert t.height == pytest.approx(15.0)
    chm_imp = CHMPlot(h_path, sep=',', impute_dbh=True)
    t_imp = chm_imp.trees[0]
    expected_diam = t_imp.get_diameter(15.0)
    assert t_imp.stemdiam == pytest.approx(expected_diam)
    assert t_imp.height == pytest.approx(15.0)

    chm2 = CHMPlot(d_path, sep=',')
    t2 = chm2.trees[0]
    assert t2.stemdiam == pytest.approx(0.30)
    assert t2.height is None
    chm2_imp = CHMPlot(d_path, sep=',', impute_h=True)
    t2_imp = chm2_imp.trees[0]
    expected_height = t2_imp.get_height(0.30)
    assert t2_imp.stemdiam == pytest.approx(0.30)
    assert t2_imp.height == pytest.approx(expected_height)


def test_tree_no_auto_impute():
    t = Tree('t', 0.0, 0.0, stemdiam_cm=30.0)
    assert t.height is None
    t2 = Tree('t2', 0.0, 0.0, height_dm=200.0)
    assert t2.stemdiam is None
