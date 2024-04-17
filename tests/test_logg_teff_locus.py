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


def main():

    df = pd.read_csv(
        join(TABLEDIR,
             'field_gyro_posteriors_20240405_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv')
    )
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
    sdf = df[sel]

    # Evaluate models up to the 10th degree as an initial range
    max_degree = 10
    evaluation_results = evaluate_models(sdf, max_degree)
    best_degree = evaluation_results.loc[np.argmin(evaluation_results.BIC), "Degree"]

    print(f'Best degree is {best_degree}')
    p, mse, coeffs = fit_polynomial(sdf, best_degree)
    print(f'Coeffs are {coeffs}')
    csvpath = join(DATADIR, "interim", "logg_teff_locus_coeffs.csv")
    df = pd.DataFrame(coeffs).T
    df.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")
    assert np.all(df.values.flatten() == coeffs)
    coeffs = pd.read_csv(csvpath).values.flatten()

    plt.close("all")
    fig, ax = plt.subplots()
    ax.scatter(sdf.adopted_Teff, sdf.adopted_logg)
    teff = np.linspace(3800,6200,500)
    ax.plot(teff, constrained_polynomial_function(teff, coeffs), zorder=5, c='r')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig('secret_test_results/constrained_polynomial_function.png')
    plt.close("all")


if __name__ == "__main__":
    main()
