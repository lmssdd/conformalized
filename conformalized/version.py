# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
#  _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "conformalized: conformalized quantile regressor for scikit-learn"
# Long description will go up on the pypi page
long_description = """

conformalized
=================

`conformalized` is a Python module for conformalized quantile regression
compatible with scikit-learn.

Please read the repository README_ on Github or our documentation_

.. _README: https://github.com/lmssdd/conformalized/blob/master/README.md

"""

NAME = "conformalized"
MAINTAINER = "Luca Massidda"
MAINTAINER_EMAIL = "luca.massidda@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/lmssdd/conformalized"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Luca Massidda"
AUTHOR_EMAIL = "luca.massidda@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__