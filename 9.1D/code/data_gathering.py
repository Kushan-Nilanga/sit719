from .utils import create_artifacts_dir
artefacts_dir = create_artifacts_dir(__file__.split('/').pop().split('.')[0])

import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)