from setuptools import setup, find_packages
import re

VERSIONFILE = "cocoa/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setup(
    name='cocoa',
    version=verstr,
    packages=find_packages(),
    license='GNU GPL V3',
    description='Toolkit for comparative connectomics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/flyconnectome/cocoa',
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='FAFB FlyWire connectomics clustering typing',
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=requirements,
    extras_require={},
    python_requires='>=3.8',
    zip_safe=False,
    include_package_data=True
)
