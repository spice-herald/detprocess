import yaml
import numpy as np
import pandas as pd

from detprocess.io._load import load_traces
from detprocess.process._features import repack_h5info_dict, SingleChannelExtractors


__all__ = [
    'process_data',
]


def _get_single_channel_feature_names(chan_dict):
    feature_list = []
    for feature in chan_dict:
        if isinstance(chan_dict[feature], dict) and chan_dict[feature]['run']:
            feature_list.append(feature)
    return feature_list

def _initialize_features(feature_list, chan, array_length):
    feature_dict = dict()
    for feat in feature_list:
        feature_dict[f'{feat}_{chan}'] = np.zeros(array_length)
    return feature_dict

def process_data(raw_file, path_to_yaml, nevents=0):
    with open(path_to_yaml) as f:
        yaml_dict = yaml.safe_load(f)

    feature_df = pd.DataFrame()

    for chan in yaml_dict:
        traces, info_dict = load_traces(
            raw_file, channels=[chan], nevents=nevents,
        )
        fs = info_dict[0]['sample_rate']
        chan_dict = yaml_dict[chan]
        template = np.loadtxt(chan_dict['template_path'])
        psd = np.loadtxt(chan_dict['psd_path'])
        feature_list = _get_single_channel_feature_names(chan_dict)
        feature_dict = _initialize_features(feature_list, chan, len(traces))

        for ii, trace in enumerate(traces[:, 0]):
            for feature_chan, feature in zip(feature_dict, feature_list):
                kwargs = {key: value for (key, value) in chan_dict[feature].items() if key!='run'}
                kwargs['template'] = template
                kwargs['psd'] = psd
                kwargs['fs'] = fs
                extractor = getattr(SingleChannelExtractors, f"extract_{feature}")
                feature_dict[feature_chan][ii] = extractor(trace, **kwargs)

        for feature in feature_dict:
            feature_df[feature] = feature_dict[feature]

    info_dict_repacked = repack_h5info_dict(info_dict)

    for info in info_dict_repacked:
        feature_df[info] = info_dict_repacked[info]

    return feature_df


