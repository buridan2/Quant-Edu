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


def random_weights(n: int) -> np.ndarray:
    """
    produces a vector of n random weights that sum to 1
    """
    weights = np.random.rand(n)
    return weights / np.sum(weights)


def random_portfolio_stats(hist_mean: pd.DataFrame, hist_cov: pd.DataFrame, weights: np.matrix) -> tuple[float, float]:
    """
    Returns the mean and standard deviation of returns for a random portfolio
    """
    mu = weights @ hist_mean.to_numpy() * 252  # annualize the returns
    sigma = np.sqrt(weights @ hist_cov.to_numpy() @ weights.T) * np.sqrt(252)  # annualize the vol

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio_stats(hist_mean=hist_mean, hist_cov=hist_cov)

    return float(mu), float(sigma)
