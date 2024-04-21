Contents:
_0_ Run the field star gyro analysis
_1_ Run the KOI lithium analysis
_2_ Build everything needed for the manuscript

Deprecated:
_X_ (Deprecated) Run the KOI gyrochronology analysis
_Y_ (Deprecated) Multiply the gyro and lithium age posteriors

--------------------
_0_ Run the field star gyro analysis:

1. `calc_field_gyro_posteriors.py`
  20230529: had S19+S21 well-defined.
  20240405: added "bonus KOIs", including Kepler1643 (and symlinked rest of 20230529)

2. `plot_field_gyro_posteriors.py`

  Makes `field_gyro_posteriors_20240405_gyro_ages_X_GDR3_S19_S21_B20.csv`, or equivalent.

3. `construct_field_star_gyro_quality_flags.py`

  Given output from #2, builds
  `field_gyro_posteriors_20240405_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv`

4. `plot_hist_field_gyro_ages.py`

--------------------
_1_ Run the KOI lithium analysis

1. Run `kepler_lithium_sample_getter.sql` on JUMP

2. `prepare_koi_jump_getter.py`

  Crossmatch the cumulative KOI sample (output of number #3 above)
  against JUMP.  Needs access to the JUMP "explore" SQL interface to
  run.  Cleans the initial SQL query results, and constucts the list of
  necessary spectra, and makes an scp script that can be used to pull
  the deblazed spectra from shrek.  Also makes CSV files like
  `/data/interim/koi_jump_getter_{sampleid}.csv`, which contain the JUMP
  and Gaia, B20, S19/21, etc information.

3. `scp_HIRES_lithium_data.sh`

  Run this to then get the HIRES spectra to the local drive.

4. `measure_Li_EWs.py`

5. `calc_koi_lithium_posteriors.py`

Use the lithium equivalent widths to calculate the lithium posteriors and cache
them to disk.

6. `plot_process_koi_lithium_posteriors.py`

Plots the results, and also makes concatenated CSV cache file:
`eagles_koi_lithium_ages_X_S19S21_dquality.csv`

--------------------
_2_ Build everything needed for the manuscript

Analysis:
  Do steps _0_ and _1_ above... then `./run_paper_scripts.sh`
todo's:
* `plot_rp_vs_age.py`
* `plot_rp_vs_porb_binage.py`
* `plot_li_vs_teff.py`


------------------------------------------
DEPRECATED ANALYSIS STEPS

--------------------
_X_ Run the KOI gyrochronology analysis

  (NOTE: this is deprecated after the field star gyro analysis approach.
  However the "fully inclusive" aspect of this initial implementation, which
  included M14/M15 knowledge, might be beneficial.  Sph>500 as a cut is also kind
  of reasonable.)

  1. `build_koi_table.py`:

  Takes the cumulative KOI table, and left-joins it against Gaia DR3, the
  Berger+20 stellar parameter catalog, and all relevant rotation period catalogs
  (McQuillan+14, Mazeh+15, Santos+19, Santos+21).  The output is written to
  `DATADIR/interim/koi_table_X_GDR3_B20_S19_S21_M14_M15.csv`

  2. `calc_koi_gyro_posteriors.py`

  Using the above table, calculate gyro the posteriors.

  3. `plot_koi_gyro_posteriors.py`

  Plot the resulting gyro posteriors, and write a table of their summary
  statistics joined against the GDR3/B20/etc data:
  `/results/koi_gyro_posteriors_{DATESTR}/step0_koi_gyro_ages_X_GDR3_B20_S19_S21_M14_M15.csv`

--------------------
_Y_ Multiply the gyro and lithium age posteriors, an entirely statistically unjustified procedure

(DEPRECATED)

  1. `calc_koi_joint_posteriors.py`

  Given the gyro and lithium posteriors, multiply them.  Saves output, including
  a merged file with all the gyro, lithium, and joint summary statistics, at 
  `koi_gyro_X_lithium_posteriors_20230116/{sampleid}_merged_joint_age_posteriors.csv`


