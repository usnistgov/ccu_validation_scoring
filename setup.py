from setuptools import setup, find_packages

try: # for pip >= 10
    import pip._internal as pip
except ImportError: # for pip <= 9.0.3
    import pip
import sys
import os
from setuptools import setup
import CCU_validation_scoring

PACKAGE_NAME = 'CCU_validation_scoring'
MINIMUM_PYTHON_VERSION = 3,8


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join('__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)

check_python_version()

setup(name=PACKAGE_NAME,
      packages=find_packages(include=['CCU_validation_scoring', 'CCU_validation_scoring.*']),
      install_requires=[
        'pandas',
        'pathlib',
        'numpy',
        'pytest',
        'matplotlib'
    ],
      version=CCU_validation_scoring.__version__,
      entry_points='''
          [console_scripts]
          CCU_scoring=CCU_validation_scoring.cli:main
      '''
      )
