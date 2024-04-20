"""
Plot "Step 0" Prot vs teff

"Step 0" means: finite teff and logg, 3800-6200K, logg>4, not FP, not grazing,
finite MES, MES>10, period not flagged by Mazeh+15.  And a period was reported
by one of M14/M15/S19/S21.

The mean Teff error is 92 K.
"""
import os
from gyrojo.paths import DATADIR, RESULTSDIR
from gyrojo.plotting import plot_koi_mean_prot_teff

outdir = os.path.join(RESULTSDIR, 'koi_mean_prot_teff')
if not os.path.exists(outdir): os.mkdir(outdir)

plot_koi_mean_prot_teff(outdir, 'koi_X_S19S21dquality', grazing_is_ok=0)
plot_koi_mean_prot_teff(outdir, 'koi_X_S19S21dquality', grazing_is_ok=1)
plot_koi_mean_prot_teff(outdir, 'deprecated_all')
