import os
from setuptools import find_packages, setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

base_packages = [
    "numpy>=1.20.3",
    "scipy>=1.6.3",
    "scikit-learn>=0.24.2",
    "pandas>=1.2.4"
]

dev_packages = []

setup(
    name="Ridge-VARX",
    version="0.0.1",
    packages=find_packages('./', exclude=["images", ".ipynb_checkpoints"]),
    description="A package for fitting a VARX model with Ridge regularization",
    long_description=read("README.md"),
    install_requires=base_packages,
    author="Stroes, F.S.D.",
    extras_require={"dev": dev_packages},
)
