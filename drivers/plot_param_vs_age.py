import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrojo.paths import PAPERDIR, RESULTSDIR
from os.path import join
import os

def plot_param_vs_age(param='koi_max_mult_ev'):

    csvpath = join(PAPERDIR, "table_allageinfo_allcols.csv")
    df = pd.read_csv(csvpath)
    sel = (
        (df.flag_is_gyro_applicable.astype(bool))
        &
        ~(df.gyro_median == '--')
    )
    df = df[sel]

    # Define the bins for gyro_median
    bins = np.arange(0, 3200, 200)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each bin
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bin_mid = bin_start + 0.5*(bin_end - bin_start)

        # Get the data for the current bin
        bin_data = (
            df[(df['gyro_median'].astype(float) >= bin_start)
               & (df['gyro_median'].astype(float) < bin_end)][param]
        )

        # Calculate the boxplot statistics
        median = np.median(bin_data)
        q1 = np.percentile(bin_data, 25)
        q3 = np.percentile(bin_data, 75)
        whisker_low = np.percentile(bin_data, 5)  # 5th percentile
        whisker_high = np.percentile(bin_data, 95)  # 95th percentile

        # Plot the boxplot
        ax.boxplot([bin_data], positions=[bin_start + 1e3*0.1],
                   widths=1e3*0.16, whis=[5, 95],
                   showfliers=False, # dont show outliers
                   boxprops=dict(color='black'),
                   whiskerprops=dict(color='black'),
                   medianprops=dict(color='red'))

        # Plot the strip plot
        ax.scatter(np.random.normal(bin_start + 100, 1e3*0.02, size=len(bin_data)), bin_data,
                   s=20, color='gray', alpha=0.5)

    # Set the x-axis tick positions and labels
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{b:.1f}' for b in bins])

    # Set the axis labels
    ax.set_xlabel('Gyro Median [Myr]')
    ax.set_ylabel(f'{param}')

    if param not in ['koi_kepmag']:
        plt.yscale('log')

    # Show the plot
    plt.tight_layout()

    outdir = join(RESULTSDIR, "pl_params_vs_age")
    if not os.path.exists(outdir): os.mkdir(outdir)
    outpath = join(outdir, f'{param}_vs_age.png')
    plt.savefig(outpath, bbox_inches='tight')

if __name__ == "__main__":
    plot_param_vs_age(param='koi_max_mult_ev')
    plot_param_vs_age(param='koi_kepmag')
    plot_param_vs_age(param='adopted_rp')
    plot_param_vs_age(param='adopted_period')
    plot_param_vs_age(param='adopted_Teff')
