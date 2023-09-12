"""
Holds all the utils functions used in the educational notebooks
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def plot_returns_dist(returns: pd.Series) -> None:
    """Plots the histogram of a series."""
    mu = returns.mean()
    sigma = returns.std()
    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 8))

    # the histogram of the data
    n, bins, patches = ax.hist(returns, 100, density=True, label="Returns")

    norm_pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)

    ax.plot(bins, norm_pdf, "--", label="Normal Distribution")
    ax.legend()
    plt.show()


def get_pdf(returns: pd.Series) -> np.ndarray:
    """Get the PDF of a returns series."""
    hist_dist = stats.rv_histogram(np.histogram(returns, bins=1000000))
    X = np.linspace(-1, 1, 1000000)
    return hist_dist.pdf(X)


def max_drawdown(returns: pd.Series) -> pd.Series:
    """Calculates the max drawdown of a returns series"""
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return drawdown
