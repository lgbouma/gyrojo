import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR
import matplotlib.pyplot as plt

# calculated in calc_field_gyro_posteriors.py
cache_id = "hist_field_gyro_ages_McQ14_20240613" # McQ14 only
datestr = 'McQ14_20240613'

cache_id = "hist_field_gyro_ages_20240530" # Santos++
datestr = '20240530'

outdir = os.path.join(RESULTSDIR, cache_id)
if not os.path.exists(outdir): os.mkdir(outdir)


# 3200 also implemented
maxages = [4000]

#no point having True; only effect is it biases against young KOIs
require_santosonly = [False]

ap.plot_hist_field_gyro_ages(
    outdir, cache_id, MAXAGE=4000, s19s21only=0, preciseagesonly=1,
    datestr=datestr
)
ap.plot_hist_field_gyro_ages(
    outdir, cache_id, MAXAGE=4000, s19s21only=False,
    datestr=datestr, dropfraclongrot=1
)
ap.plot_hist_field_gyro_ages(
    outdir, cache_id, MAXAGE=4000, s19s21only=False,
    datestr=datestr
)

