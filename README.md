#  `detprocess`: Detector processing code for feature extraction

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

`detprocess` is a Python package meant for feature extraction from raw detector data saved by [`pytesdaq`](https://github.com/berkeleytes/pytesdaq). The main functionality of the code is contained in `detprocess/process`, which contains all the possible features to be extracted and the general pipeline of how features are extracted. This package also contains helper IO functions for loading events from `pytesdaq` and saving the processed data as Pandas DataFrames.

## Installation

To install the most recent development version of RQpy, clone this repo, then from the top-level directory of the repo, type the following line into your command line

`pip install .`

If using a shared Python installation, you may want to add the `--user` flag to the above line. Note the package requirements, especially the need for [QETpy](https://github.com/ucbpylegroup/QETpy) and [`pytesdaq`](https://github.com/berkeleytes/pytesdaq).
