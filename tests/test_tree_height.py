import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trees import Stand


def build_stand(tmp_path):
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
    return Stand(1, csv_path, mapping=mapping, sep=',', impute_dbh=True, impute_h=True)


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


def test_stand_filters_invalid_stand_ids(tmp_path):
    data = [
        {
            'Stand': 1,
            'PLOT': 1,
            'TreeID': 'valid-1',
            'X_GROUND': 0.0,
            'Y_GROUND': 0.0,
            'STEMDIAM': 25.0,
            'H': 12.0,
        },
        {
            'Stand': 'abc',  # invalid stand id should be skipped
            'PLOT': 1,
            'TreeID': 'invalid',
            'X_GROUND': 1.0,
            'Y_GROUND': 1.0,
            'STEMDIAM': 26.0,
            'H': 13.0,
        },
        {
            'Stand': 1,
            'PLOT': 1,
            'TreeID': 'valid-2',
            'X_GROUND': 2.0,
            'Y_GROUND': 2.0,
            'STEMDIAM': 27.0,
            'H': 14.0,
        },
        {
            'Stand': 2,  # different stand should be filtered out
            'PLOT': 2,
            'TreeID': 'other-stand',
            'X_GROUND': 3.0,
            'Y_GROUND': 3.0,
            'STEMDIAM': 28.0,
            'H': 15.0,
        },
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'trees_with_invalid.csv'
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

    with pytest.warns(UserWarning, match=r"Skipping 1 row\(s\) with non-numeric stand IDs"):
        stand = Stand(1, csv_path, mapping=mapping, sep=',')

    assert len(stand.plots) == 1
    plot = stand.plots[0]
    assert [tree.tree_id for tree in plot.trees] == ['valid-1', 'valid-2']
