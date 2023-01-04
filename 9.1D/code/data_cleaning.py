import pandas as pd
import matplotlib.pyplot as plt
from .utils import create_artifacts_dir
artefacts_dir = create_artifacts_dir(__file__.split('/').pop().split('.')[0])


def get_na_count(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def save_histogram_plots(df: pd.DataFrame, n_rows, n_cols) -> None:
    # plot all histograms in one figure
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))
    for i, col in enumerate(df.columns):
        ax = axes[i // n_cols, i % n_cols]
        df[col].hist(ax=ax, bins=20)
        ax.set_title(col)

    fig.savefig(artefacts_dir + '/histograms.png')
