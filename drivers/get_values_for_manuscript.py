import os
from os.path import join
import numpy as np, pandas as pd
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data
)
from gyrojo.paths import DATADIR
from gyrojo.papertools import update_latex_key_value_pair as ulkvp
from gyrojo.papertools import read_latex_key_value_pairs

##########################################
# Stars with rotation periods and quality flags calculated
df = get_gyro_data(
    "Santos19_Santos21_dquality", drop_grazing=0, drop_highruwe=0
)

assert pd.isnull(df.Prot).sum() == 0
assert len(np.unique(df.KIC)) == len(df)

# Number of unique stars with Santos+ rotation period reported
N = len(df)
ulkvp('nuniqstarsantosrot', N)

# Number of unique stars with Santos+ rotation period reported, and
# 3800<Teff<6200K (i.e. can compute a gyro age).
sel = (df.adopted_Teff > 3800) & (df.adopted_Teff < 6200)
df = df[sel]
N = len(df)
ulkvp('nuniqstarsantosrotteffcut', N)

# Number of unique stars with Santos+ rotation period reported, and
# 3800<Teff<6200K (i.e. can compute a gyro age).
sel = (
      (~df['flag_logg'])
      #&
      #(~df['flag_ruwe_outlier'])
      &
      (~df['flag_dr3_non_single_star'])
      &
      (~df['flag_camd_outlier'])
      #&
      #(df['flag_not_CP_CB'])
      &
      (~df['flag_in_KEBC'])
      &
      (df['adopted_Teff'] > 3800)
      &
      (df['adopted_Teff'] < 6200)
)
df = df[sel]
N = len(df)
ulkvp('nuniqstarsantosallbutruwe', N)

df = get_gyro_data(
    "Santos19_Santos21_dquality", drop_grazing=0, drop_highruwe=1
)

# Number of unique stars with Santos+ rotation, and gyro applicable
N = len(df[df.flag_is_gyro_applicable])
ulkvp('nuniqstarsantosrotgyroappl', N)

##########################################
# get PLANET numbers!
kdf = get_koi_data('cumulative-KOI', drop_grazing=0)

# Number of non-false positive KOIs from cumulative koi df
N = len(kdf[~kdf.flag_koi_is_fp])
ulkvp('nnonfpcumkois', N)

# Number of unique non-false positive KOI host stars from cumulative koi df
N = len(np.unique(kdf[~kdf.flag_koi_is_fp].kepid))
ulkvp('nnonfpcumkoihosts', N)

##########################################
# ...with age results
df, _, _ = get_age_results(
    whichtype='gyro_li', drop_grazing=0, drop_highruwe=0
)
assert pd.isnull(df.gyro_median).sum() == 0

# Number of planets with a gyro age, including grazing & high ruwe cases
N = len(df)
ulkvp('nplwgyroagewithgrazingandhighruwe', N)

# Number of planet-hosting stars with a gyro age, including grazing & high ruwe cases
N = len(np.unique(df.kepid_x))
ulkvp('nplhoststarwgyroagewithgrazingandhighruwe', N)

N = len(np.unique(df[df.flag_ruwe_outlier].kepid_x))
ulkvp('nplhoststarwgyroagejusthighruwe', N)


# Number of planets with a gyro age, including grazing & high ruwe cases, below 1Gyr
sel = (df.gyro_median > 0) & (df.gyro_median <= 1000)
N = len(df[sel])
ulkvp('nplyounggyro', N)
N = len(np.unique(df[sel].kepid_x))
ulkvp('nplhostsyounggyro', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, 1-2 Gyr
sel = (df.gyro_median > 1000) & (df.gyro_median <= 2000)
N = len(df[sel])
ulkvp('nplmidgyro', N)
N = len(np.unique(df[sel].kepid_x))
ulkvp('nplhostsmidgyro', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, 2-3 Gyr
sel = (df.gyro_median > 2000) & (df.gyro_median <= 3000)
N = len(df[sel])
ulkvp('nploldgyro', N)
N = len(np.unique(df[sel].kepid_x))
ulkvp('nplhostsoldgyro', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, below 1Gyr at 2sigma
sel = (df['gyro_median'] + df['gyro_+2sigma']  <= 1000)
N = len(df[sel])
ulkvp('nplyounggyrotwosigma', N)
N = len(np.unique(df[sel].kepid_x))
ulkvp('nplhostsyounggyrotwosigma', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, below 1Gyr at 3sigma
sel = (df['gyro_median'] + df['gyro_+3sigma']  <= 1000)
N = len(df[sel])
ulkvp('nplyounggyrothreesigma', N)
N = len(np.unique(df[sel].kepid_x))
ulkvp('nplhostsyounggyrothreesigma', N)

# ... now drop grazing cases, keeping high ruwe
df, _, _ = get_age_results(
    whichtype='gyro', drop_grazing=1, drop_highruwe=0
)

# Number of planets with a gyro age, dropping grazing cases
N = len(df)
ulkvp('nplwgyroagenograzing', N)

# Number of planet-hosting stars with a gyro age, dropping grazing cases
N = len(np.unique(df.kepid_x))
ulkvp('nplhoststarwgyroagenograzing', N)

##########################################
# assertion checks:

d = read_latex_key_value_pairs()

# lithium pieces written by koi_jump_getter; others written here.
# they should be the same, else the two refer to different samples.
assert d['nlithiumgyrostars'] == d['nplhoststarwgyroagewithgrazingandhighruwe']
assert d['nlithiumgyroplanets'] == d['nplwgyroagewithgrazingandhighruwe']
