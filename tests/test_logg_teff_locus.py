"""
This script defines a locus in logg vs adopted teff that can be used to
separate "evolved" from "non-evolved" stars:

....
"We explored various options, and found that the $+1$$\sigma$ relative
isochronal uncertainty reported by \citet{Berger_2020a_catalog},
$+\sigma_{t,{\rm B20,iso}}/t_{\rm B20,iso}$, reasonably separated FGK stars
``near'' and ``far'' from the main-sequence.  We therefore selected stars with
$t_{\rm B20,iso}/(+\sigma_{t,{\rm B20,iso}})\approx 3$, and fitted an an
$N^{\rm th}$-order polynomial in the $\log g$-$T_{\rm eff}$ plane to these
stars between 4100 and 5800\,K.  We let the order of the fit vary from $N$=1 to
10, and minimizing the Bayesian Information Criterion yielded $N$=6.  The
resulting locus is shown in Figure~\ref{fig:stellarprops}."
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_squared_error
from numpy import array as nparr

from gyrojo.paths import TABLEDIR, DATADIR
from gyrojo.locus_definer import (
    detect_outliers_windowed, fit_polynomial, bic, evaluate_models,
    constrained_polynomial_function
)
from os.path import join

def complex_selection(df):
    # Define the Teff range and corresponding rel_E_Age range
    teff_min, teff_max = 5400, 6000
    rel_E_age_min, rel_E_age_max = 0.34, 0.5

    # Calculate the slope and intercept of the linear function
    slope = (rel_E_age_max - rel_E_age_min) / (teff_max - teff_min)
    intercept = rel_E_age_min - slope * teff_min

    # Create a boolean mask for the selection criteria
    sel = (
        (df.adopted_Teff >= 3700) &
        (df.adopted_Teff <= 5900) &
        (df.adopted_logg > 4.2) &
        (df.adopted_logg < 4.63) &
        (
            (
                (df.adopted_Teff < teff_min) &
                (df.b20t2_rel_E_Age > 0.33) &
                (df.b20t2_rel_E_Age < 0.35)
            ) |
            (
                (df.adopted_Teff >= teff_min) &
                (df.adopted_Teff <= teff_max) &
                (df.b20t2_rel_E_Age > (df.adopted_Teff * slope + intercept - 0.01)) &
                (df.b20t2_rel_E_Age < (df.adopted_Teff * slope + intercept + 0.01))
            )
        )
    )

    return sel

def simple_selection(df):

    sel = (
        (df.b20t2_rel_E_Age > 0.33)
        &
        (df.b20t2_rel_E_Age < 0.35)
        &
        (df.adopted_Teff > 3700)
        &
        (df.adopted_Teff < 5900)
        &
        (df.adopted_logg > 4.2)
        &
        (df.adopted_logg < 4.63)
    )
    return sel


def main(selfn=None):

    assert selfn in ['simple', 'complex', 'manual']

    if selfn in ['simple', 'complex']:
        df = pd.read_csv(
            join(TABLEDIR,
                 'field_gyro_posteriors_20240405_gyro_ages_'
                 'X_GDR3_S19_S21_B20_with_qualityflags.csv')
        )
        if selfn == 'simple':
            sel = simple_selection(df)
        else:
            sel = complex_selection(df)
        sdf = df[sel]

    elif selfn == 'manual':
        df = pd.read_csv(
            join(DATADIR, "interim", "koi_logg_teff_near_main_sequence.csv")
        )
        sdf = df

    # Evaluate models up to the 10th degree as an initial range
    max_degree = 10
    evaluation_results = evaluate_models(sdf, max_degree)
    best_degree = evaluation_results.loc[np.argmin(evaluation_results.BIC), "Degree"]

    print(f'Best degree is {best_degree}')
    p, mse, coeffs = fit_polynomial(sdf, best_degree)
    print(f'Coeffs are {coeffs}')
    csvpath = join(DATADIR, "interim", f"logg_teff_locus_coeffs_{selfn}.csv")
    df = pd.DataFrame(coeffs).T
    df.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")
    assert np.all(df.values.flatten() == coeffs)
    coeffs = pd.read_csv(csvpath).values.flatten()

    plt.close("all")
    fig, ax = plt.subplots()
    ax.scatter(sdf.adopted_Teff, sdf.adopted_logg)
    teff = np.linspace(3800,6200,500)
    ax.plot(teff, constrained_polynomial_function(teff, coeffs, selfn=selfn),
            zorder=5, c='r')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig(f'secret_test_results/constrained_polynomial_function_{selfn}.png')
    plt.close("all")


if __name__ == "__main__":
    # Idea: fit locus to things near σiso/tiso ~= 0.33
    main(selfn='simple')
    # Idea: fit locus to things near σiso/tiso ~= 0.33, but vary the cut for
    # hotter stars to get it closer to where it makes sense
    main(selfn='complex')
    # Idea: keep simple for the cool stars.  For the hot stars, just go based
    # off where you see things coming close/far from "main sequence" of actual
    # KOIs (manually selected and trimmed in glue)
    main(selfn='manual')
