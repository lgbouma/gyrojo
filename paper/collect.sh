r="/Users/luke/Dropbox/proj/gyrojo/results"

# plot_gyromodeldispersion
cp $r/gyromodeldispersion/gyromodeldispersion.pdf .

# plot_st_params
# ...CAMD:
cp $r/st_params/st_params_M_G_vs_dr3_bp_rp.pdf .
cp $r/st_params/st_params_M_G_vs_adopted_Teff.pdf .
# ...logg vs Teff
cp $r/st_params/st_params_adopted_logg_vs_adopted_Teff.pdf .
# ...G vs Teff
cp $r/st_params/st_params_dr3_phot_g_mean_mag_vs_adopted_Teff.pdf .

# plot_star_Prot_Teff
cp $r/star_Prot_Teff/prot_teff_Santos19_Santos21_dquality.pdf .

# plot_koi_mean_prot_teff
cp $r/koi_mean_prot_teff/koi_mean_prot_teff_koi_X_S19S21dquality_keepgrazing.pdf .

# plot_hist_field_gyro_ages 
cp $r/hist_field_gyro_ages_20240405/hist_samples_koi_gyro_ages_hist_field_gyro_ages_20240405_maxage3200.pdf .

# plot_li_vs_teff
cp $r/li_vs_teff/li_vs_teff_koi_X_S19S21dquality_eagles_logy.pdf .
cp $r/li_vs_teff/li_vs_teff_koi_X_JUMP_eagles_logy.pdf .
cp $r/li_vs_teff/li_vs_teff_koi_X_S19S21dquality_eagles_showpoints_nodata.pdf .

# planets
r="/Users/luke/Dropbox/proj/NEA_age_plots/results/rp_vs_period_scatter"
cp $r/rp_vs_period_scatter_20240415_colorbyage_showaux-gyro_anyyoung_anyyoung.pdf .


########## 
# appendices (probably omit)
r="/Users/luke/Dropbox/proj/gyrojo/results"
cp $r/gyroage_vs_teff/gyroage_vs_teff_errs_showplanets_linear.png .
