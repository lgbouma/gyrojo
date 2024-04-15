#!/bin/bash

python make_gyroonly_table.py
cwd=$(pwd)
cd ~/Dropbox/proj/NEA_age_plots/drivers
python plot_rp_vs_period_scatter.py
cd "$cwd"

echo 'done'
exit 0

python plot_star_Prot_Teff.py
python plot_koi_mean_prot_teff.py
python plot_hist_field_gyro_ages.py
python plot_li_vs_teff.py
python plot_camd.py

python get_values_for_manuscript.py
python make_younghighlight_table.py


