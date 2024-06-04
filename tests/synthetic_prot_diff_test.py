import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from gyrojo.prot_uncertainties import get_empirical_prot_uncertainties
from aesthetic.plot import set_style, savefig
from gyrojo.paths import RESULTSDIR

def generate_observed_periods(true_periods, noise_func):
    observed_periods = true_periods + np.random.normal(0, noise_func(true_periods))
    return observed_periods

# Generate "true" rotation periods from a truncated normal distribution
mean, sd, lower, upper = 18, 8, 1, np.inf
true_periods = truncnorm.rvs((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd, size=10000)

# Generate two sets of "observed" rotation periods with Gaussian noise
P1 = generate_observed_periods(true_periods, get_empirical_prot_uncertainties)
P2 = generate_observed_periods(true_periods, get_empirical_prot_uncertainties)

# Create a DataFrame with the observed periods
df = pd.DataFrame({'P1': P1, 'P2': P2})

# Calculate the absolute difference between P1 and P2 and add it as a new column
df['abs_diff'] = np.abs(df['P1'] - df['P2'])

plt.close("all")
set_style("clean")
fig, axs = plt.subplots(ncols=2, figsize=(5,2.5))

for ix, ax in enumerate(axs):
    ax.scatter(df['P1'], df['abs_diff'], c='k', s=0.5, linewidths=0)

    for pct in np.arange(1, 6, 1):
        ax.plot([0, 50], (pct / 100) * np.array([0, 50]), lw=0.5, label=f"{pct}%")

    _prot = np.linspace(0.1, 50, 1000)
    prot_err = get_empirical_prot_uncertainties(_prot)
    ax.plot(_prot, prot_err, lw=2, color='lime', label='σProt')

    # Bin the data and calculate median and ±1 sigma values
    bins = np.arange(0, df['P1'].max() + 1, 1)
    binned_data = pd.cut(df['P1'], bins)
    grouped_data = df.groupby(binned_data)
    y_medians = grouped_data['abs_diff'].median()
    y_q16 = grouped_data['abs_diff'].quantile(0.16)
    y_q84 = grouped_data['abs_diff'].quantile(0.84)

    # Overplot the median and ±1 sigma values
    x_mids = (bins[:-1] + bins[1:]) / 2
    x_err = np.ones_like(x_mids)
    y_err = np.vstack([y_medians - y_q16, y_q84 - y_medians])
    ax.errorbar(x_mids, y_medians, xerr=x_err, yerr=y_err, fmt='o',
                color='lightgray', elinewidth=0.5, capsize=0, lw=0,
                mew=0.5, markersize=1)

    ax.set_xlabel("P1 [day]")
    if ix == 0:
        ax.set_ylabel("|P1 - P2| [day]")
        ax.set_ylim([-0.2, 2.2])
    if ix == 1:
        ax.legend(fontsize='small')
        ax.set_ylim([-0.2, 10])

    ax.set_xlim([-2,52])

outdir = join(RESULTSDIR, "reinhold2023_exploration")
if not os.path.exists(outdir): os.mkdir(outdir)
outpath = join(outdir, f'synthetic.png')

savefig(fig, outpath, writepdf=0)
