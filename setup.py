from setuptools import setup, find_packages

setup(
    name="ft4ftirs",
    version="1.0.0",
    author="Jon Gabirondo-López",
    author_email="jon.gabirondol@ehu.eus",
    description="A package for processing FTIR interferograms.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jongablop/ft4ftirs",
    license="GPL-3.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "brukeropusreader>=1.3.0"
    ],
    extras_require={
        # FER export/import (ft4ftirs.io.fer) is optional; the rest of the
        # package does not import ferpy.
        "fer": ["ferpy"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
