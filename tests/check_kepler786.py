import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# Download the Kepler light curve for Kepler-786
search_result = lk.search_lightcurve('Kepler-786', author='Kepler', cadence='long')
lc_collection = search_result.download_all()

# Iterate over each quarter
for lc in lc_collection:
    # Median-normalize the light curve
    lc = lc.normalize(unit='ppm')
    
    # Create a plot for the current quarter
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lc.time.value, lc.flux.value, 'k.', markersize=1)
    ax.set_xlabel('Time - 2454833 (days)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Kepler-786 - Quarter {lc.quarter}')
    
    # Save the plot to disk
    plt.tight_layout()
    plt.savefig(f'secret_test_results/kepler_786_q{str(lc.quarter).zfill(3)}.png', dpi=300)
    plt.close()

# Stitch all quarters together
lc_stitched = lc_collection.stitch()

# Sigma clip outliers
lc_clipped = lc_stitched.remove_outliers(sigma=5)

# Median-normalize the light curve
lc_normalized = lc_clipped.normalize(unit='ppm')

# Create a plot for the stitched and normalized light curve
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(lc_normalized.time.value, lc_normalized.flux.value, 'k.', markersize=1)
ax.set_xlabel('Time - 2454833 (days)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Kepler-786 - Stitched and Normalized Light Curve')
plt.tight_layout()
plt.savefig('secret_test_results/kepler_786_stitched_normalized.png', dpi=300)
plt.close()

# Run Lomb-Scargle periodogram
frequency, power = LombScargle(lc_normalized.time.value, lc_normalized.flux.value).autopower()
period = 1 / frequency

# Create a plot for the periodogram
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(period, power, 'k-')
ax.set_xlabel('Period (days)')
ax.set_ylabel('Power')
ax.set_title('Kepler-786 - Lomb-Scargle Periodogram')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('secret_test_results/kepler_786_periodogram.png', dpi=300)
plt.close()

# Find the top five periods
top_indices = power.argsort()[-20:][::-1]
top_periods = period[top_indices]

print("Top Five Periods:")
ix = 0
for i, period_val in enumerate(top_periods, start=1):
    if ix >= 5:
        break
    if period_val > 0.1:
        print(f"{i}. {period_val:.4f} days")
        ix += 1
    else:
        continue
