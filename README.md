#  `detprocess`: Detector processing code for feature extraction

[![PyPI](https://img.shields.io/pypi/v/detprocess)](https://pypi.org/project/detprocess/) [![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

`detprocess` is a Python package meant for feature extraction from raw detector data saved by [`pytesdaq`](https://github.com/berkeleytes/pytesdaq). The main functionality of the code is contained in `detprocess.process`, which contains all the possible features to be extracted and the general pipeline of how features are extracted. This package also contains helper IO functions for loading events from `pytesdaq` and saving the processed data as Vaex DataFrames.

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Available Base Features](#available-base-features)
  - [YAML File](#yaml-file)
  - [Extract features](#extracting-features)
## Installation

To install the most recent release of `detprocess`, type the following line into your command line

`pip install detprocess --upgrade`

To install the most recent development version of `detprocess`, clone this repo, then from the top-level directory of the repo, type the following line into your command line

`pip install .`

If using a shared Python installation, you may want to add the `--user` flag to the above line. Note the package requirements, especially the need for [QETpy](https://github.com/ucbpylegroup/QETpy) and [`pytesdaq`](https://github.com/berkeleytes/pytesdaq).

## Usage

One of the goals of this package is to keep the feature extraction pipeline simple and modular. The pipeline in mind can be approximated as follows:
1. Know what features you want to extract, see: [Available Features](#available-base-features)
2. Create a YAML file specifying feature extraction options, see: [YAML File](#yaml-file)
3. Run the feature extraction code on your data, see: [Extracting Features](#extracting-features)



### Available Base Features

The available features to extract are stored as the static methods of `detprocess.FeatureExtractors`. Each of these methods take your data and extract that specific feature from each event.

At this time, the available features are:

 - `of1x1_nodelay`: returns the no delay optimum filter amplitude and chi-square (as in, the amplitude if the template is not allowed a time degree-of-freedom)
 - `of1x1_unconstrained`: returns the unconstrained optimum filter amplitude, time offset, and chi-square
 - `of1x1_constrained`: returns the constrained optimum filter amplitude, time offset, and chi-square, where a window constraint is specified
 - `baseline`: returns the average value from the beginning of an event up to some specified index
 - `integral`: returns the integral of the event **(no baseline subtraction)** from some specified start index to some specified end index
 - `maximum`: returns the maximum value of the event from some specified start index to some specified end index
 - `minimum`: returns the minimum value of the event from some specified start index to some specified end index
 - `energyabsorbed`: returns the energy absorbed by a Transition-Edge Sensor (TES) based on the inputted parameters that correspond to the TES bias point


The base features can be used to define new features in the configuration with different settings, for example "baseline_pre" defined in a yaml file (see below) can use the "baseline" based algorithm.

More features can be added either locally, or if there is a feature that is universally useful, then please submit a Pull Request! 

There are also features that are stored directly in `pytesdaq` files, which `detprocess` will also extract. These are:

- `event_number`
- `event_index`
- `dumpn_umber`
- `series_number`
- `event_time`
- `trigger_type`
- `trigger_amplitude`
- `trigger_time`

To understand these more, we direct the user to [`pytesdaq`'s Documentation](https://github.com/berkeleytes/pytesdaq).

### YAML File

The YAML file contains nearly all of the information needed to extract features from your data. This is done on purpose, as it allows the user to easily reuse/change their YAML files for different processing, to easily version control their processing, and easily share their processing setup with collaborators. To make sure we can do this, the YAML must  have a specific format. Here's an example below.

```yaml
filter_file: ./filter_example.hdf5
detector1:
    o1x1_nodelay:
        run: True
    of1x1_unconstrained:
        run: False
    of1x1_constrained:
        run: False
        window_min_from_trig_usec: -400
        window_max_from_trig_usec: 400
   of1x1_constrained_glitch
        run: True
        window_min_from_trig_usec: -400
        window_max_from_trig_usec: 400
        base_algorithm: of1x1_unconstrained
        template_tag: glitch	
    baseline_pre:
        run: True
        base_algorithm: baseline
        window_min_from_start_usec: 0
        window_max_from_trig_usec: -1000
    integral:
        run: True
        start_index: 0
        window_min_from_trig_usec: -500
        window_max_from_trig_usec: 500
```

In this YAML file, we first specify the filter file, which contains the PSD and templates for each channels. The pulse template should be a single array that contains the expected pulse shape, normalized to have a pulse amplitude of 1 and have a baseline of 0. The current-referenced PSD should be a single array that contains the two-sided PSD in units of $\mathrm{A}^2/\mathrm{Hz}$. Note that both of these will should have the same digitization rate and/or length as the data that will be processed to be able to calculate the optimum filter features. We must then specify which channel will be processed, in this case `detector1`. This should match the channel name in the corresponding `pytesdaq` file. the optimum filter features.

We have also specified to extract different features from each event: `of1x1_nodelay`, `baseline`, and `integral`. This is done by specifying `run: True` in the file, as compared to `run: False` for `of1x1_unconstrained` and `of1x1_constrained`. Note that it is fine to simple exclude features from the YAML file, as they simply will not be calculated (e.g. `energyabsorbed` is not included in this example).


### Extracting Features

See notebook detprocess/examples/run_detprocess.ipynb