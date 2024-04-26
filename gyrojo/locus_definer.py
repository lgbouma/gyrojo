"""
Contents:
    detect_outliers_windowed
    fit_polynomial
    bic
    evaluate_models
    constrained_polynomial_function

(contents are ~90% chatgpt4/claude3 generated)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_squared_error
from numpy import array as nparr

from gyrojo.paths import TABLEDIR, DATADIR
from os.path import join

def detect_outliers_windowed(x, y, window_size=10, threshold=2):
    """Detect outliers in a dataset using a windowed-slider approach.

    Args:
        x (array-like): The x-values of the dataset.
        y (array-like): The y-values of the dataset.
        window_size (int, optional): The size of the sliding window.
            Default is 10.
        threshold (float, optional): The threshold for identifying outliers
            in terms of standard deviations. Default is 2.

    Returns:
        array: A boolean array indicating whether each point is an outlier
            (True) or not (False).
    """
    # Convert x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Initialize an array to store outlier flags
    outliers = np.full(len(x), False)

    # Iterate over the data points using a sliding window
    for i in range(len(x) - window_size + 1):
        # Extract the current window
        window_x = x[i:i+window_size]
        window_y = y[i:i+window_size]

        # Calculate the mean and standard deviation of y values in the window
        mean_y = np.mean(window_y)
        std_y = np.std(window_y)

        # Identify outliers within the window based on the threshold
        window_outliers = np.abs(window_y - mean_y) > threshold * std_y

        # Update the outlier flags for the current window
        outliers[i:i+window_size] |= window_outliers

    return outliers


def fit_polynomial(data, N):
    """
    Fit an Nth-order polynomial to the given data points.

    Args:
    data (DataFrame): The data points to fit.
    N (int): The order of the polynomial to fit.

    Returns:
    tuple: The polynomial coefficients and the mean squared error of the fit.
    """

    xvals, yvals = nparr(data['adopted_Teff']), nparr(data['adopted_logg'])
    inds = np.argsort(xvals)
    xvals, yvals = xvals[inds], yvals[inds]

    outliers = detect_outliers_windowed(
        xvals, yvals, window_size=15, threshold=1.5
    )
    xvals, yvals = xvals[~outliers], yvals[~outliers]

    # Fit the polynomial
    coeffs = np.polyfit(xvals, yvals, N)
    p = np.poly1d(coeffs)

    # Calculate the mean squared error
    mse = mean_squared_error(data['adopted_logg'], p(data['adopted_Teff']))

    return p, mse, coeffs


def bic(mse, n, k):
    """
    Calculate the Bayesian Information Criterion for a model.

    Args:
    mse (float): The mean squared error of the model.
    n (int): The number of data points.
    k (int): The number of parameters in the model.

    Returns:
    float: The BIC score.
    """
    return n * np.log(mse) + k * np.log(n)


def evaluate_models(data, max_degree):
    """
    Evaluate polynomial models of various degrees and calculate their BIC scores.

    Args:
    data (DataFrame): The data points to fit.
    max_degree (int): The maximum degree of the polynomial to test.

    Returns:
    DataFrame: A dataframe containing the degree, MSE, and BIC score for each model.
    """
    results = []
    for N in range(1, max_degree + 1):
        p, mse, _ = fit_polynomial(data, N)
        k = N + 1  # Number of parameters is degree + 1 (including the constant term)
        bic_score = bic(mse, len(data), k)
        results.append({
            'Degree': N,
            'MSE': mse,
            'BIC': bic_score
        })

        # Plot the data and the fitted polynomial
        plt.figure(figsize=(10, 6))
        plt.scatter(data['adopted_Teff'], data['adopted_logg'], s=10,
                    label='Data Points')
        plt.plot(np.sort(data['adopted_Teff']),
                 p(np.sort(data['adopted_Teff'])), color='red',
                 label=f'Polynomial Degree {N}')
        plt.xlabel('Adopted_Teff')
        plt.ylabel('Adopted_logg')
        plt.title(f'Polynomial Degree {N} Fit')
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.savefig(f'secret_test_results/polynomial_degree_{N}.png')
        plt.close()

    return pd.DataFrame(results)


def constrained_polynomial_function(temperatures, coeffs, selfn=None):
    """
    Vectorized function to evaluate the constrained polynomial function over an
    array of temperatures.

    Args:
        temperatures (np.ndarray): Array of temperature values where the logg
        should be evaluated.

        coeffs (np.ndarray): Coefficients of the polynomial.

    Returns:
        np.ndarray: Array of logg values with constraints applied.
    """
    assert selfn in ['simple', 'complex', 'manual']

    if isinstance(temperatures, float):
        temperatures = np.array([temperatures])

    # Calculate polynomial values
    poly_vals = np.polyval(coeffs, temperatures)

    logg_vals = poly_vals*1.

    if selfn  in ['simple','complex']:
        # Apply the floor of 4.25 to logg values
        logg_vals = np.maximum(poly_vals, 4.25)

        logg_vals = np.where(
            (temperatures > 5800),
            4.25,
            logg_vals
        )

        logg_vals = np.where(
            (temperatures < 3800) | (temperatures > 6200),
            np.nan,  # Replace 'None' with 'np.nan' for array operations
            logg_vals
        )

        if len(temperatures[(temperatures > 3800) & (temperatures < 4100)]) > 0:
            logg_vals = np.where(
                (temperatures > 3800) & (temperatures < 4100),
                np.min(logg_vals[(temperatures > 3800) & (temperatures < 4200)]),
                logg_vals
            )

    else:

        logg_vals = np.where(
            (temperatures < 3800) | (temperatures > 6200),
            np.nan,  # Replace 'None' with 'np.nan' for array operations
            logg_vals
        )

    return logg_vals
