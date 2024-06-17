import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR

# calculated in calc_field_gyro_posteriors.py
cache_id = "hist_field_gyro_ages_20240530" # Santos++
datestr = '20240530'

cache_id1 = "hist_field_gyro_ages_McQ14_20240613" # McQ14 only
datestr1 = 'McQ14_20240613'

outdir = os.path.join(
    RESULTSDIR,
    f'mcq14_vs_santos_age_histograms_Santos{datestr}_vs_McQ{datestr1}'
)
if not os.path.exists(outdir): os.mkdir(outdir)


# 3200 also implemented
maxages = [4000]

#no point having True; only effect is it biases against young KOIs
require_santosonly = [False]

for maxage in maxages:
    ap.plot_hist_field_gyro_ages(
        outdir, cache_id, MAXAGE=maxage, s19s21only=0, preciseagesonly=1,
        datestr=datestr, cache_id1=cache_id1, datestr1=datestr1
    )

for maxage in maxages:
    for s19s21only in require_santosonly:
        ap.plot_hist_field_gyro_ages(
            outdir, cache_id, MAXAGE=maxage, s19s21only=s19s21only,
            datestr=datestr, cache_id1=cache_id1, datestr1=datestr1
        )
