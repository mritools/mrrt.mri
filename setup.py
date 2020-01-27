#!/usr/bin/env python
import os
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

PACKAGES = find_packages()

# Get version and release info, which is all stored in mrrt/mri/version.py
ver_file = os.path.join("mrrt", "mri", "version.py")
with open(ver_file) as f:
    exec(f.read())
# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ["setuptools >= 24.2.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


cmdclass = {"test": PyTest}

opts = dict(
    name=NAME,
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
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    install_requires=REQUIRES,
    python_requires=PYTHON_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    requires=REQUIRES,
    cmdclass=cmdclass,
)

if __name__ == "__main__":
    setup(**opts)
