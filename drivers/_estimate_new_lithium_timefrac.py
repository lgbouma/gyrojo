import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle
from os.path import join

from gyrojo.paths import DATADIR, LOCALDIR, RESULTSDIR
from gyrojo.getters import get_gyro_data

csvpath = join(
    DATADIR, "raw",
    "20240407_JUMP_kepler_lithium_sample_no_iodine_constraint.csv"
)

from astropy.time import Time

time_strings = [i[:-6] for i in jdf['utctime'].tolist()]

times = Time(time_strings, format='iso', scale='utc')

# berger+2018 submission date
t0 = Time('2017-09-29 11:11:11')

N_new = len(times[times.jd > t0.jd])
N_total = len(times)

print(N_new / N_total)
