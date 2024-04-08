-- This script is a preliminary getter for the Kepler lithium sample.
-- It is to be executed on JUMP.
-- The string-matching portion will also grab K2, KELT, KH-15D, etc.
-- The cross-match against my N=100 gyro list is done
-- offline, because I don't know how to upload remote tables.
-- Only the i chip is needed, based on the 6708A Li feature.
-- We also only really need 1 spectrum per star.  Ideal
-- would be to combine to get highest possible S/N.  But
-- that will be a future script.
-- It also produces a bunch of duplicates, but those are easier
-- cleaned in pandas, which has a cleaner remove duplicates function.
-- (We only need the highest count spectrum per star for now!)
SELECT DISTINCT db_star.name,
       db_star.ra,
       db_star.dec,
       db_observation.instrument,
       db_observation.utctime,
       db_observation.observation_id,
       db_observation.id as obs_id,
       db_observation.isjunk,
       db_spectrum.filename,
       db_spectrum.chip,
       db_spectrum.spectrum_type,
       db_observation_hires_j.exposure_time,
       db_observation_hires_j.counts
FROM db_star
JOIN db_observation ON (db_star.name = db_observation.star_id)
JOIN db_spectrum on (db_observation.id = db_spectrum.observation_id)
JOIN db_observation_hires_j on (db_spectrum.observation_id=db_observation_hires_j.observation_id)
WHERE (
  (
  UPPER(db_star.name) LIKE 'CK%'
  OR
  UPPER(db_star.name) LIKE 'K%'
  OR
  UPPER(db_star.name) LIKE 'KEPLER%'
  OR
  UPPER(db_star.name) LIKE 'KIC%'
  )
--  AND
--  db_observation_hires_j.iodine_in = False
  AND
  db_observation.isjunk = False
  AND
  db_spectrum.chip = 'i'
  AND
  db_spectrum.spectrum_type = 'deblazed'
)
ORDER BY db_star.name, db_observation_hires_j.counts DESC
