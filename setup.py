#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rxntp",
    version="0.0.1",
    author="Xiaorui Dong",
    author_email="xiaorui@mit.com",
    description="A Python package for reaction template analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiaoruiDong/RXNTP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    license = "MIT License",
    python_requires='>=3.7',
)
