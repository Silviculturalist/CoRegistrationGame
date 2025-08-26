import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from chm_plot import CHMPlot


def test_skip_rows_without_height_or_dbh(tmp_path):
    data = [
        {'X': 0.0, 'Y': 0.0, 'IDALS': 't1', 'DBH': 30.0},
        {'X': 1.0, 'Y': 1.0, 'IDALS': 't2', 'DBH': ''},
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'chm.csv'
    df.to_csv(csv_path, index=False)

    plot = CHMPlot(csv_path, sep=',')

    assert len(plot.trees) == 1
    assert plot.trees[0].tree_id == 't1'
