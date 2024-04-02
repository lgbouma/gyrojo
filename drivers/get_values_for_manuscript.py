import numpy as np, pandas as pd
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data
)
from gyrojo.papertools import update_latex_key_value_pair as ulkvp

##########################################
# Stars with rotation periods and quality flags calculated
df = get_gyro_data("Santos19_Santos21_dquality")

assert pd.isnull(df.Prot).sum() == 0
assert len(np.unique(df.KIC)) == len(df)

# Number of unique stars with Santos+ rotation period reported
N = len(df)
ulkvp('nuniqstarsantosrot', N)

# Number of unique stars with Santos+ rotation period reported, and
# 3800<Teff<6200K (i.e. can compute a gyro age).
sel = (df.adopted_Teff >= 3800) & (df.adopted_Teff <= 6200)
df = df[sel]
N = len(df)
ulkvp('nuniqstarsantosrotteffcut', N)

# Number of unique stars with Santos+ rotation, and gyro applicable
N = len(df[df.flag_is_gyro_applicable])
ulkvp('nuniqstarsantosrotgyroappl', N)

##########################################
# get PLANET numbers!
kdf = get_koi_data('cumulative-KOI')

# Number of non-false positive KOIs from cumulative koi df
N = len(kdf[~kdf.flag_koi_is_fp])
ulkvp('nnonfpcumkois', N)

# Number of unique non-false positive KOI host stars from cumulative koi df
N = len(np.unique(kdf[~kdf.flag_koi_is_fp].kepid))
ulkvp('nnonfpcumkoihosts', N)

##########################################
# ...with age results
df, _, _ = get_age_results(whichtype='gyro', drop_grazing=0)

# Number of planets with a gyro age, including grazing cases
N = len(df)
ulkvp('nplwgyroage', N)

# Number of planet-hosting stars with a gyro age, including grazing cases
N = len(np.unique(df.kepid_x))
ulkvp('nplhoststarwgyroage', N)

df, _, _ = get_age_results(whichtype='gyro', drop_grazing=1)

# Number of planets with a gyro age, dropping grazing cases
N = len(df)
ulkvp('nplwgyroagenograzing', N)

# Number of planet-hosting stars with a gyro age, dropping grazing cases
N = len(np.unique(df.kepid_x))
ulkvp('nplhoststarwgyroagenograzing', N)
