from __future__ import print_function
import sys, os
import warnings
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import numpy
except ImportError:
    warnings.warn('numpy is required during installation', ImportWarning)

try:
    import scipy
except ImportError:
    warnings.warn('scipy is required during installation', ImportWarning)

# Get version and release info, which is all stored in conformalized/version.py
ver_file = os.path.join('conformalized', 'version.py')
with open(ver_file) as f:
    exec(f.read())

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=find_packages(),
            install_requires=INSTALL_REQUIRES)

if __name__ == '__main__':
    setup(**opts)
    