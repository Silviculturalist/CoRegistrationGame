import pandas as pd
import numpy as np
import sys
from pathlib import Path
import types

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal stub for pynput to avoid GUI dependencies during import
mock_pynput = types.ModuleType("pynput")
mock_keyboard = types.ModuleType("keyboard")
mock_pynput.keyboard = mock_keyboard
sys.modules.setdefault("pynput", mock_pynput)
sys.modules.setdefault("pynput.keyboard", mock_keyboard)

from trees import Plot, Tree
from app import App


def test_transformation_serialization_primitives(tmp_path):
    plot = Plot(plotid=1)
    plot.append_tree(Tree(1, 0.0, 0.0, stemdiam_cm=1.0, height_dm=10.0))
    plot.append_tree(Tree(2, 1.0, 0.0, stemdiam_cm=1.0, height_dm=10.0))
    plot.append_tree(Tree(3, 0.0, 1.0, stemdiam_cm=1.0, height_dm=10.0))

    app = App.__new__(App)
    app.plot_transformations = {}
    app.store_transformations(plot)

    df = pd.DataFrame.from_dict(app.plot_transformations, orient="index")
    csv_path = tmp_path / "transforms.csv"
    df.to_csv(csv_path, index=False)
    df_loaded = pd.read_csv(csv_path)

    # Ensure primitive columns exist and contain numeric data
    for col in ["tx", "ty", "r00", "r01", "r10", "r11"]:
        assert col in df_loaded.columns
        assert np.issubdtype(df_loaded[col].dtype, np.number)

    # Identity rotation and zero translation for unmodified plot
    assert np.isclose(df_loaded.loc[0, "tx"], 0.0)
    assert np.isclose(df_loaded.loc[0, "ty"], 0.0)
    assert np.isclose(df_loaded.loc[0, "r00"], 1.0)
    assert np.isclose(df_loaded.loc[0, "r01"], 0.0)
    assert np.isclose(df_loaded.loc[0, "r10"], 0.0)
    assert np.isclose(df_loaded.loc[0, "r11"], 1.0)
