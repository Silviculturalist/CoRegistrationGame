import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from chm_plot import CHMPlot
from trees import Plot, Tree


def test_skip_rows_without_height_or_dbh(tmp_path):
    data = [
        {"X": 0.0, "Y": 0.0, "IDALS": "t1", "DBH": 30.0},
        {"X": 1.0, "Y": 1.0, "IDALS": "t2", "DBH": ""},
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "chm.csv"
    df.to_csv(csv_path, index=False)

    plot = CHMPlot(csv_path, sep=",")

    assert len(plot.trees) == 1
    assert plot.trees[0].tree_id == "t1"


def test_remove_matches_without_height(tmp_path):
    chm_df = pd.DataFrame(
        [
            {"X": 0.5, "Y": 0.5, "IDALS": "c1", "DBH": 30.0},
        ]
    )
    chm_csv = tmp_path / "chm.csv"
    chm_df.to_csv(chm_csv, index=False)

    chm_plot = CHMPlot(chm_csv, sep=",")

    field_plot = Plot(1)
    field_plot.append_tree(Tree("t1", 0.0, 0.0, height_dm=200.0))  # 20 m height

    chm_plot.remove_matches(field_plot)

    assert len(chm_plot.trees) == 0
    assert len(chm_plot.removed_stems[-1]) == 1
