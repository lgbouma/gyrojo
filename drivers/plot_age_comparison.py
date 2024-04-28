import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'age_comparison')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_age_comparison(outdir, ratio_v_gyro=1)
ap.plot_age_comparison(outdir, iso_v_gyro=1, logscale=0)
ap.plot_age_comparison(outdir, iso_v_gyro=1, logscale=1)
ap.plot_age_comparison(outdir, hist_age_unc_ratio=1)
