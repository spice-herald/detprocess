#  `detprocess`: Detector processing code for feature extraction

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

`detprocess` is a Python package meant for feature extraction from raw detector data saved by [`pytesdaq`](https://github.com/berkeleytes/pytesdaq). The main functionality of the code is contained in `detprocess/process`, which contains all the possible features to be extracted and the general pipeline of how features are extracted. This package also contains helper IO functions for loading events from `pytesdaq` and saving the processed data as Pandas DataFrames.

## Installation

To install the most recent development version of RQpy, clone this repo, then from the top-level directory of the repo, type the following line into your command line

`pip install .`

If using a shared Python installation, you may want to add the `--user` flag to the above line. Note the package requirements, especially the need for [QETpy](https://github.com/ucbpylegroup/QETpy) and [`pytesdaq`](https://github.com/berkeleytes/pytesdaq).

## Usage

One of the goals of this package is to keep the feature extraction pipeline simple and modular. The pipeline in mind can be approximated as follows:
1. Know what features you want to extract, see: [Available Features](#available-features)
2. Create a YAML file specifying feature extraction options, see: [YAML File](#yaml-file)
3. Run the feature extraction code on your data
4. Analyze the features that you have extracted


### Available Features

The available features to extract are stored as the static methods of `detprocess.SingleChannelExtractors`. Each of these methods take your data and extract that specific feature from each event. At this time, the available features are:
 - `of_nodelay`
 - `of_unconstrained`
 - `of_constrained`
 - `baseline`
 - `integral`
 - `maximum`
 - `minimum`
 - `energyabsorbed`


### YAML File

The YAML file contains nearly all of the information needed to extract features from your data. This is done on purpose, as it allows the user to easily reuse/change their YAML files for different processing, to easily version control their processing, and easily share their processing setup with collaborators. To make sure we can do this, the YAML must  have a specific format. Here's an example below.

```yaml
detector1:
    template_path: /path/to/template.txt
    psd_path: /path/to/psd.txt
    of_nodelay:
        run: True
    of_unconstrained:
        run: False
    of_constrained:
        run: False
        windowcenter: 0
        nconstrain: 100
    baseline:
        run: True
        end_index: 16000
    integral:
        run: True
        start_index: 0
        end_index: 16000
```
