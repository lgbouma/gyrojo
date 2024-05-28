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

def main(selfn=None):

    assert selfn in ['poly']

    df = pd.read_csv(
        join(TABLEDIR,
             'field_gyro_posteriors_20240405_gyro_ages_'
             'X_GDR3_S19_S21_B20_with_qualityflags.csv')
    )

    sdf = df[
        (df.nkoi >= 1)
        &
        (df.dr3_bp_rp > 0.5)
        &
        (df.dr3_bp_rp < 3)
        &
        (df.M_G < 12)
        &
        (df.M_G > 0)
    ]

    # Evaluate models up to the 10th degree as an initial range
    max_degree = 10
    evaluation_results = evaluate_models(sdf, max_degree, xkey='dr3_bp_rp', ykey='M_G')
    best_degree = evaluation_results.loc[np.argmin(evaluation_results.BIC), "Degree"]

    print(f'Best degree is {best_degree}')
    p, mse, coeffs = fit_polynomial(
        sdf, best_degree, xkey='dr3_bp_rp', ykey='M_G'
    )
    print(f'Coeffs are {coeffs}')
    csvpath = join(DATADIR, "interim", f"MG_bprp_locus_coeffs_{selfn}.csv")
    df = pd.DataFrame(coeffs).T
    df.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")
    assert np.all(df.values.flatten() == coeffs)
    coeffs = pd.read_csv(csvpath).values.flatten()

    plt.close("all")
    fig, ax = plt.subplots()
    ax.scatter(sdf.dr3_bp_rp, sdf.M_G, s=1, c='k')

    bprp = np.linspace(0.45, 3, 500)
    poly_vals = np.polyval(coeffs, bprp)
    ax.plot(bprp, poly_vals, zorder=5, c='r')
    ax.plot(bprp, poly_vals-1.5, zorder=5, c='r', alpha=0.5)
    ax.plot(bprp, poly_vals+1, zorder=5, c='r', alpha=0.5)

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig(f'secret_test_results/MG_vs_bprp_constrained_polynomial_function_{selfn}.png')
    plt.close("all")


if __name__ == "__main__":
    main(selfn='poly')
