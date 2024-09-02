from setuptools import setup, find_packages

setup(
    name="ft4ftirs",
    version="0.1.0",
    author="Jon Gabirondo-López",
    author_email="jon.gabirondol@ehu.eus",
    description="A package for processing FTIR interferograms.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jongablop/ft4ftirs",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "brukeropusreader>=1.3.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
