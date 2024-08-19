"""
This module contains paths that are re-used throughout the project.  Crucially,
it makes a hidden local cache directory, at ~/.gyrointerp_cache, where results
from large batch runs will by default be stored.
"""
import os
from os.path import join
from gyrojo import __path__
__path__ = list(__path__)

DATADIR = join(__path__[0], 'data')

RESULTSDIR = join(os.path.dirname(__path__[0]), 'results')

TABLEDIR = join(RESULTSDIR, 'tables')

CACHEDIR = join(os.path.expanduser('~'), '.gyrojo_cache')

PAPERDIR = join(os.path.dirname(__path__[0]), 'papers', 'paper')

for l in [DATADIR, CACHEDIR, RESULTSDIR, TABLEDIR]:
    if not os.path.exists(l):
        print(f"Making {l}")
        os.mkdir(l)

# used for making the manuscript's plots, and not any reusable functionality
LOCALDIR = join(os.path.expanduser('~'), 'local')
