#  `detprocess`: Detector processing code for feature extraction

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

`detprocess` is a Python package meant for feature extraction from raw detector data saved by [`pytesdaq`](https://github.com/berkeleytes/pytesdaq). The main functionality of the code is contained in `detprocess.process`, which contains all the possible features to be extracted and the general pipeline of how features are extracted. This package also contains helper IO functions for loading events from `pytesdaq` and saving the processed data as Pandas DataFrames.

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Available Features](#available-features)
  - [YAML File](#yaml-file)
  - [Extracting Features](#extracting-features)
  - [Loading Features](#loading-features)
- [Advanced Usage](#advanced-usage)
  - [Adding New Features](#adding-new-features)
  - [High-Performance Computing](#high-performance-computing)

## Installation

To install the most recent release of `detprocess`, type the following line into your command line

`pip install detprocess`

To install the most recent development version of `detprocess`, clone this repo, then from the top-level directory of the repo, type the following line into your command line

`pip install .`

If using a shared Python installation, you may want to add the `--user` flag to the above line. Note the package requirements, especially the need for [QETpy](https://github.com/ucbpylegroup/QETpy) and [`pytesdaq`](https://github.com/berkeleytes/pytesdaq).

## Usage

One of the goals of this package is to keep the feature extraction pipeline simple and modular. The pipeline in mind can be approximated as follows:
1. Know what features you want to extract, see: [Available Features](#available-features)
2. Create a YAML file specifying feature extraction options, see: [YAML File](#yaml-file)
3. Run the feature extraction code on your data, see: [Extracting Features](#extracting-features)
4. Analyze the features that you have extracted, see: [Loading Features](#loading-features)


### Available Features

The available features to extract are stored as the static methods of `detprocess.SingleChannelExtractors`. Each of these methods take your data and extract that specific feature from each event. At this time, the available features are:

 - `of_nodelay`: returns the no delay optimum filter amplitude and chi-square (as in, the amplitude if the template is not allowed a time degree-of-freedom)
 - `of_unconstrained`: returns the unconstrained optimum filter amplitude, time offset, and chi-square
 - `of_constrained`: returns the constrained optimum filter amplitude, time offset, and chi-square, where a window constraint is specified
 - `baseline`: returns the average value from the beginning of an event up to some specified index
 - `integral`: returns the integral of the event **(no baseline subtraction)** from some specified start index to some specified end index
 - `maximum`: returns the maximum value of the event from some specified start index to some specified end index
 - `minimum`: returns the minimum value of the event from some specified start index to some specified end index
 - `energyabsorbed`: returns the energy absorbed by a Transition-Edge Sensor (TES) based on the inputted parameters that correspond to the TES bias point

More features can be added either locally, or if there is a feature that is universally useful, then please submit a Pull Request! See [Advanced Usage](#advanced-usage) for instructions on adding features.

There are also features that are stored directly in `pytesdaq` files, which `detprocess` will also extract. These are:

- `eventnumber`
- `eventindex`
- `dumpnumber`
- `seriesnumber`
- `eventtime`
- `triggertype`
- `triggeramp`
- `triggertime`

To understand these more, we direct the user to [`pytesdaq`'s Documentation](https://github.com/berkeleytes/pytesdaq).

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

In this YAML file, we first specify which channel will be processed, in this case `detector1`. This should match the channel name in the corresponding `pytesdaq` file. We have supplied absolute paths to both the pulse template and the current-referenced power spectral density (PSD) to be used by the optimum filters. The pulse template should be a single array that contains the expected pulse shape, normalized to have a pulse amplitude of 1 and have a baseline of 0. The current-referenced PSD should be a single array that contains the two-sided PSD in units of <img src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BA%7D%5E2%20%2F%20%5Cmathrm%7BHz%7D">. Note that both of these will should have the same digitization rate and/or length as the data that will be processed to be able to calculate the optimum filter features.

We have also specified to extract 3 different features from each event: `of_nodelay`, `baseline`, and `integral`. This is done by specifying `run: True` in the file, as compared to `run: False` for `of_unconstrained` and `of_constrained`. Note that it is fine to simple exclude features from the YAML file, as they simply will not be calculated (e.g. `energyabsorbed` is not included in this example).

For the features themselves, the functions to extract them may require extra arguments. In this example, `baseline` requires the user to pass the `end_index` value, and `integral` requires the user to pass both `start_index` and `end_index`. There are no default values for these features, so these arguments must be passed. To know what is needed for each feature, the docstrings of these features can be accessed through `detprocess.SingleChannelExtractors`.

### Extracting Features

Feature extraction is meant to be very easy once we have our features and our YAML file. Essentially all of the work is done by `detprocess.process_data`. This function takes the absolute path to the file to be processed and the absolute path to the YAML file, and then will return a DataFrame containing all of the extracted features. The user can ensure these are automatically saved to a folder of their choice by also passing a path to the `savepath` optional argument. An example of the expected workflow is shown in [`examples/run_detprocess.ipynb`](examples/run_detprocess.ipynb).

### Loading Features

After feature extraction is complete (assuming that the user chose to save the processed data), it is simple to load the features. As part of the IO functionality, `detprocess` has a function to load the extracted features into a Pandas DataFrame. This is done by `detprocess.io.load_features`, the example notebook [`examples/run_detprocess.ipynb`](examples/run_detprocess.ipynb) shows usage of this function.

## Advanced Usage

Beyond the basic functionality of `detprocess`, advanced users may want to do more, such as adding new features or interfacing with a high-performance computing (HPC) cluster. In this section, we detail how to do these.

### Adding New Features

For advanced users, there may be a need to add new features for extraction which are not currently included. To do this, the user must add a new feature as a `staticmethod` in `detprocess.SingleChannelExtractors`. To understand how to do this, let's look at a couple of the currently-supported features. Below we show an excerpt of the feature definitions, showing how `of_nodelay` and `integral` are defined in `detprocess/process/_features.py`.

```python
class SingleChannelExtractors(object):
    """
    A class that contains all of the possible feature extractors
    for a given trace, assuming processing on a single channel.
    Each feature extraction function is a staticmethod for processing
    convenience.

    """

    @staticmethod
    def of_nodelay(trace, template, psd, fs, **kwargs):
        """
        Feature extraction for the no delay Optimum Filter.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature from.
        template : ndarray
            The template to use for the optimum filter.
        psd : ndarray
            The PSD to use for the optimum filter.
        fs : float
            The digitization rate of the data in trace.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        OF = qp.OptimumFilter(
            trace,
            template,
            psd,
            fs,
        )
        ofamp_nodelay, ofchi2_nodelay = OF.ofamp_nodelay()

        retdict = {
            'ofamp_nodelay': ofamp_nodelay,
            'ofchi2_nodelay': ofchi2_nodelay,
        }
        return retdict

## features between these have been truncated from the source code

    @staticmethod
    def integral(trace, start_index, end_index, fs, **kwargs):
        """
        Feature extraction for the pulse integral.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature from.
        start_index : int
            The index of the trace to start the integration.
        end_index : int
            The index of the trace to end the integration.
        fs : float
            The digitization rate of the data in trace.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        integral = np.trapz(trace[start_index:end_index]) / fs

        retdict = {
            'integral': integral,
        }

        return retdict

```

We should note that, when comparing to our example [YAML file](#yaml-file), there are three special arguments that are being passed to the features, which we did not directly define in that file: `template`, `psd`, and `fs`. The reason is that `template` and `psd` are defined from opening the text files `template_path` and `psd_path`, while `fs` is contained in the `pytesdaq` file being processed, so it does not need to be defined by the user. These special arguments can always be included in a new feature by directly specifying them, otherwise they are ignored. To support these special features, each extraction function **must** have `**kwargs` in its definition.

For the `integral` feature, we see that there are two arguments which are defined in our example [YAML file](#yaml-file): `start_index` and `end_index`. As these are arguments that are needed specifically for this feature, they must be defined in the YAML file.

Each feature function can return multiple values, if needed, as shown by `of_nodelay`. The `detprocess` processing routine will use the defined keys in the returned dictionary `retdict` to save them to the outputted Pandas DataFrame. Note that, to ensure unique names if we are processing multiple channels in the same file, `detprocess` will append the channel name to the saved feature. That is, the outputted DataFrame for our example [YAML file](#yaml-file) for these two specified feature function would have the following keys saved: `ofamp_nodelay_detector1`, `ofchi2_nodelay_detector1`, and `integral_detector1`.

To create a new feature, we would follow the format of the above features. Let's create a feature called `example_feature`, defined below, where `some_function` takes the place of the code that would be used to extract the features. Let's say this feature needs the parameters `param1` and `param2`, as well as the special argument `fs`. We include the required `**kwargs` for support of the special features.

```python
    @staticmethod
    def example_feature(trace, param1, param2, fs, **kwargs):
        """
        Feature extraction for the pulse integral.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature from.
        param1 : float
            The first parameter to pass to our feature extraction.
        param2 : float
            The first parameter to pass to our feature extraction.
        fs : float
            The digitization rate of the data in trace.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        output1, output2 = some_function(trace, param1, param2, fs)

        retdict = {
            'example_feature1': output1,
            'example_feature2': output2,
        }

        return retdict

```

This feature must be added as one of the static methods in the class `SingleChannelExtractors`, located in `detprocess/process/_features.py`. After adding the code, the user must then recompile the package via `pip install .`, and the feature can be extracted by `detprocess`.

If we added this feature to our example [YAML file](#yaml-file), the corresponding entry should be of the form (where we are currently passing dummy values to the dummy parameters):

```yaml
    example_feature:
        run: True
        param1: 0
        param2: 1
```

When running `detprocess.process_data`, the outputted Pandas DataFrame would then have the keys `example_feature1_detector1` and `example_feature2_detector1`, which can then be accessed by the user in their analysis.

### High-Performance Computing

Detailing this usage is a work-in-progress.
