import yaml
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import vaex as vx
import importlib
import sys
import os
from glob import glob
from pprint import pprint
from multiprocessing import Pool
from itertools import repeat
from datetime import datetime
import stat
import time
import astropy
import copy
from humanfriendly import parse_size
from itertools import groupby
from detprocess.core import Noise, DIDVAnalysis, FilterData, Template, NoiseModel
from detprocess import utils
import pytesdaq.io as h5io
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')


__all__ = [
    'FilterDataProcessing'
]


class FilterDataProcessing:
    """
    Class to manage stand alone dIdV (such as Beginning of Run dIdV) 
    and noise PSD processing
    """

    def __init__(self, files_or_path, series=None,
                 restricted=False,
                 config_file=None,
                 config_dict=None,
                 facility=None,
                 verbose=True):
        """
        Intialize data processing 
        
        Parameters
        ---------
    
        files_or_path : str or list of str
           File(s) or path(s) containing randoms, contiunuous and/or dIdV data
           Only data from single group allowed
                      
        config_file : str 
           Full path and file name to the YAML settings for the
           processing.
            

        verbose : bool, optional
            if True, display info

        Return
        ------
        None
        """

        # display
        self._verbose = verbose

        # Raw file list
        data = self._get_file_list(files_or_path, series=series,
                                   restricted=restricted)
        
        raw_notrig, raw_exttrig, raw_thresh, base_path, group_name =  data

        self._raw_data_notrig  = raw_notrig
        self._raw_data_exttrig = raw_exttrig
        self._raw_data_threshtrig = raw_thresh
        self._group_name = group_name
        self._base_path  = base_path
     
        # detector config
        data_notrig = self._get_data_config(raw_notrig)
        self._series_config_notrig = data_notrig[0]
        self._det_config_notrig = data_notrig[1]
        self._channels_notrig  = data_notrig[2]
        self._is_continuous = self._is_continuous(raw_notrig)
        
        data_exttrig = self._get_data_config(raw_exttrig)
        self._series_config_exttrig = data_exttrig[0]
        self._det_config_exttrig = data_exttrig[1]
        self._channels_exttrig  = data_exttrig[2]
        
        data_threshtrig = self._get_data_config(raw_thresh)
        self._series_config_threshtrig = data_threshtrig[0]
        self._det_config_threshtrig = data_threshtrig[1]
        self._channels_threshtrig  = data_threshtrig[2]

        # read config file
        available_channels =  (self._channels_notrig
                               + self._channels_exttrig
                               + self._channels_threshtrig)
        available_channels = list(set(available_channels))

        self._processing_config = None
        
        if config_file is not None:
            self._processing_config = self._read_config(config_file,
                                                        available_channels)
        elif config_dict is not None:
            self._processing_config = copy.deepcopy(config_dict)
        else:
            raise ValueError('ERROR: Processing configuration required! '
                             'Either "config_file" or "config_dict" argument')

        # filter data to store results
        self._filter_data = FilterData()


        # facility
        if facility is None:

            data_dict = dict()
            if self._series_config_notrig:
                data_dict = self._series_config_notrig
            elif self._series_config_exttrig:
                data_dict = self._series_config_exttrig
            elif self._series_config_threshtrig:
                data_dict = self._series_config_threshtrig
            
            for key, item in data_dict.items():
                if 'facility' in item:
                    facility = item['facility']
                break
                    
        if facility is None:
            raise ValueError('ERROR: unable to find facility number! Add '
                             '"facility" argument!')

        self._facility = facility


        # sample rate
        self._sample_rate = None

        
        
    def check_config(self, processing_type):
        """
        Check configuration file
        """

        if (not self._processing_config or
            self._processing_config is None):
            raise ValueError('ERROR: Processing config not found!')


        if processing_type == 'didv':
            
            # let's do some preliminary checks
            if not self._raw_data_exttrig:
                raise ValueError('ERROR: Unable to process dIdV. No '
                                 'dIdV data found!')
            
            if 'didv' not in self._processing_config:
                raise ValueError(f'ERROR: Input yaml file does '
                                 f'not contain didv processing'
                                 f'configurations!')
            
            # didv config
            didv_config = self._processing_config['didv']

            # channels
            channels = list(didv_config.keys())
            
            # check IVsweep results available
            for chan in channels:
                
                if 'bias_parameter_type' not in didv_config[chan]:
                    raise ValueError(
                        f'ERROR: "bias_parameter_type" parameters '
                        f'required to process dIdV for  channel {chan}')
                    
                if ('ivsweep_file' not in didv_config[chan]
                    and 'ivsweep_data' not in didv_config[chan]):
                    raise ValueError(
                        f'ERROR: I0 required for channel '
                        f'{chan}, either "ivsweep_file"   or '
                        f'"ivsweep_data" (dictionary)')

        elif processing_type == 'noise':
            
            if not self._raw_data_notrig:
                raise ValueError('ERROR: Unable to process noise. No '
                                 'randoms or continuous  data found!')

            if (self._processing_config is not None
                and 'noise' not in self._processing_config):
                raise ValueError(f'ERROR: Input yaml file does '
                                 f'not contain noise processing '
                                 f'configurations!')

            # noise config
            noise_config = self._processing_config['noise']

            # channels
            channels = list(noise_config.keys())
            
            for chan in channels:

                tag_list = list()
                
                # check byseries parameters
                if 'calc_noise_byseries' in noise_config[chan]:
                    
                    if 'byseries_noise_tag' not in noise_config[chan]:
                        raise ValueError(
                            f'ERROR: Unable to process noise. Argument '
                            f'"byseries_noise_tag" is missing for channel '
                            f'{chan}!')
                
                    tag_list.append(noise_config[chan]['byseries_noise_tag'])
                    
                # check global parameters
                if 'calc_noise_global' in noise_config[chan]:

                    if 'global_noise_tag_list' not in noise_config[chan]:
                        raise ValueError(
                            f'ERROR: Unable to process noise. Argument '
                            f'"global_noise_tag_list" is missing for '
                            f'channel {chan}!')
                    
                    tag_list.extend(noise_config[chan]['global_noise_tag_list'])

                # check tags
                tag_list = list(set(tag_list))

                # check tags
                for tag in tag_list:

                    if (tag not in noise_config[chan]
                        and self._is_continuous):
                        raise ValueError(
                            f'ERROR: No information found for '
                            f'tag {tag}, channel {chan}. '
                            f'Trace length (msec or samples) is '
                            f'required to process noise of '
                            f'continuous data!')
                        
        elif processing_type == 'template':
            
            if ('template' not in self._processing_config):
                raise ValueError(f'ERROR: Input yaml file does '
                                 f'not contain template generation '
                                 f'configurations!')

            # template config
            template_config = self._processing_config['template']

            # channels
            channels = list(template_config.keys())
            
            # check all channels
            for chan in channels:

                if chan not in self._processing_config['template']:
                    raise ValueError(f'ERROR: No template configuration '
                                     f'for channel {chan}')

                chan_config = self._processing_config['template'][chan]

                if 'template_tag_list' not in chan_config:
                    raise ValueError(f'ERROR: "template_tag_list" not found '
                                     f'for channel {chan}. Either add parameter '
                                     f'or disable channel!')

                for tag in chan_config['template_tag_list']:
                    if tag not in chan_config:
                        raise ValueError(f'ERROR: No configuration found for '
                                         f'tag {tag}, channel {chan}')
                    else:
                        if 'template_poles' not in chan_config[tag]:
                            raise ValueError(f'ERROR: No "template_poles" parameter '
                                             f'found for tag {tag}, channel {chan}')
                        else:
                           template_poles = chan_config[tag]['template_poles']
                           if ('amplitude_A' not in chan_config[tag]
                               or 'rise_time' not in chan_config[tag]
                               or ' fall_time_1' not in chan_config[tag]):
                               raise ValueError(
                                   f'ERROR: Missing template parameters '
                                   f'for tag {tag}, channel {chan}. Check '
                                   f'example for details.')
                           
  
    def proces_didv(self,
                    channels=None,
                    processing_config=None,
                    processing_id=None,
                    lgc_output=False,
                    lgc_save=False,
                    save_file_path=None,
                    ncores=1):
        """
        Processing dIdV
        """

        self.process(channels=channels,
                     enable_didv=True,
                     processing_config=processing_config,
                     processing_id=processing_id,
                     lgc_output=lgc_output,
                     lgc_save=lgc_save,
                     save_file_path=save_file_path,
                     ncores=ncores)


    def proces_noise(self,
                     channels=None,
                     processing_config=None,
                     processing_id=None,
                     lgc_output=False,
                     lgc_save=False,
                     save_file_path=None,
                     ncores=1):
        """
        Processing Noise
        """

        self.process(channels=channels,
                     enable_noise=True,
                     processing_config=processing_config,
                     processing_id=processing_id,
                     lgc_output=lgc_output,
                     lgc_save=lgc_save,
                     save_file_path=save_file_path,
                     ncores=ncores)

            
    def process(self,
                channels=None,
                enable_noise=False,
                enable_didv=False,
                enable_template=False,
                nevents=None,
                processing_id=None,
                lgc_output=False,
                lgc_save=False,
                save_file_path=None,
                ncores=1):
        
        """
        Process data 

        Parameters
        ---------
        
        channels : str or list of str, optional
          detector channel(s)
          if None: use all available channels
        

        lgc_save : bool, optional
           if True, save dataframe in hdf5 files
           Default: False

        lgc_output : bool, optional
           if True, return dataframe 
           Default: False

        save_file_path : str, optional
           base directory where output group will be saved
           default: same base path as input data
    
        ncores: int, optional
           number of cores that will be used for processing
           default: 1
        """
        
        # check enable IV, dIdV
        if not enable_didv and not enable_noise and not enable_template:
            raise ValueError('ERROR: You need to enable dIdV, psd, '
                             'and or template calculation!')

        # check config
        if enable_didv:
            self.check_config('didv')

        if enable_noise:
            self.check_config('noise')
        
        if enable_template:
            self.check_config('template')   
                           
        # check if file or path
        if (save_file_path is not None
            and not (os.path.isfile(save_file_path)
                     or os.path.isdir(save_file_path))):
            raise ValueError('ERROR: "save_file_path" argument '
                             'should be a file or a path!')

        # let's check channels to be processed
        channels_didv = list()
        channels_noise = list()
        channels_template = list()
        
        if channels is not None:
            
            if isinstance(channels, str):
                channels = [channels]

            # check
            for chan in channels:
                
                if enable_didv:
                                                 
                    if chan not in self._channels_exttrig:
                        raise ValueError(
                            f'ERROR: no {chan} available in dIdV raw data!'
                        )
                    if chan not in self._processing_config['didv'].keys():
                        raise ValueError(
                            f'ERROR: no {chan} available in didv configuration data!'
                        )
                
                if enable_noise:

                    channels_split, sep = utils.split_channel_name(
                        chan, self._channels_notrig)
                           
                    for chan_split in channels_split:
                        if chan_split not in self._channels_notrig:
                            raise ValueError(
                                f'ERROR: no {chan_split} available in raw data!'
                            )
                    if chan not in self._processing_config['noise'].keys():
                        raise ValueError(
                            f'ERROR: no {chan} available in noise configuration data!'
                        )

                if enable_template:
                     if chan not in self._processing_config['template'].keys():
                        raise ValueError(
                            f'ERROR: no {chan} available in template configuration data!'
                        )
            channels_didv = channels.copy()
            channels_noise = channels.copy()
            channels_template = channels.copy()
                
        else:

            #  get channels from config
            channels_didv = list(self._processing_config['didv'].keys())
            channels_noise =  list(self._processing_config['noise'].keys())
            channels_template =  list(self._processing_config['template'].keys())

        # intialize output
        output_dict = {'didv':dict(),
                       'noise': dict(),
                       'template': dict()}


        self._processing_id = processing_id
        
        # ================================
        # dIdV processing
        # ================================
        
        if enable_didv:
            
            # number of cores
            nseries = len(self._det_config_exttrig.keys())
            if ncores > nseries:
                ncores = nseries
                
            # series list 
            series_list = list(self._raw_data_exttrig.keys())

            # didv config
            didv_config = self._processing_config['didv']

            for chan in channels_didv:
                
                # configuration dictionary
                chan_config = copy.deepcopy(didv_config[chan])

                # default auto_infinite_lgain
                if 'auto_infinite_lgain' not in chan_config:
                    chan_config['auto_infinite_lgain'] = True

                # filter data tag
                filterdata_tag = 'default'
                if 'didv_tag' in chan_config:
                    filterdata_tag = chan_config['didv_tag']
                
                # intitialize output
                output_df = None

                if ncores == 1:
                    output_df = self._process_didv(1,
                                                   series_list,
                                                   chan,
                                                   chan_config,
                                                   nevents)
                else:
            
                    # split data
                    series_list_split = self._split_series(series_list,
                                                           ncores)
                    
                    # lauch pool processing
                    if self._verbose:
                        print(f'INFO: Processing with be split '
                              f'between {ncores} cores!')
            
                    node_nums = list(range(ncores+1))[1:]
                    pool = Pool(processes=ncores)
                    output_df_list = pool.starmap(self._process_didv,
                                                  zip(node_nums,
                                                      series_list_split,
                                                      repeat(chan),
                                                      repeat(chan_config),
                                                      repeat(nevents)))
                    pool.close()
                    pool.join()

                    # concatenate output
                    df_list = list()
                    for df in output_df_list:
                        if df is not None:
                            df_list.append(df)
                    if df_list:
                        output_df = pd.concat(df_list)

                # add in filter_data
                metadata = {'processing_id': self._processing_id,
                            'group_name': self._group_name}
                
                self._filter_data.set_didv_dataframe(chan, output_df,
                                                     metadata=metadata,
                                                     tag=filterdata_tag)

                # store in dictionary
                output_dict['didv'][chan] = {'df': output_df}


        # ================================
        # Noise processing
        # ================================
        
        if enable_noise:

            # number of cores
            nseries = len(self._det_config_notrig.keys())
            if ncores > nseries:
                ncores = nseries

            # series list 
            series_list = list(self._raw_data_notrig.keys())

            # noise config
            noise_config = self._processing_config['noise']

            # loop channels and launch processing
            for chan in channels_noise:

                # configuration dictionary
                chan_config = copy.deepcopy(noise_config[chan])

                # Set default if needed
                # check byseries parameters

                tag_list = list()
                
                if 'calc_noise_byseries' not in chan_config:
                    chan_config['calc_noise_byseries'] = False
                else:
                    tag_list.append(chan_config['byseries_noise_tag'])
                                                            
                    # psd amps  
                    if 'psd_amp_freq_range_list' not in chan_config:
                        chan_config['psd_amp_freq_range_list'] = None

                    if 'corr_coeff_freq_range_list' not in chan_config:
                        chan_config['corr_coeff_freq_range_list'] = None
                    
                # check global parameters
                if 'calc_noise_global' not in chan_config:
                    chan_config['calc_noise_global'] = False
                else:
                    tag_list.extend(chan_config['global_noise_tag_list'])

                # check tags
                tag_list = list(set(tag_list))

                for tag in tag_list:

                    if tag not in chan_config:
                        chan_config[tag] = dict()
                        
                    if 'trace_length_msec' not in  chan_config[tag]:
                        chan_config[tag]['trace_length_msec'] = None
                    if 'trace_length_samples' not in  chan_config[tag]:
                        chan_config[tag]['trace_length_samples'] = None
                    if 'pretrigger_length_msec' not in  chan_config[tag]:
                        chan_config[tag]['pretrigger_length_msec'] = None
                    if 'pretrigger_length_samples' not in  chan_config[tag]:
                        chan_config[tag]['pretrigger_length_samples'] = None
                        
                # calc noise psd by series
                if  chan_config['calc_noise_byseries']:

                    # tag
                    filterdata_tag = chan_config['byseries_noise_tag']
                                                           
                    # intitialize output
                    output_df = None
                    
                    if ncores == 1:
                        output_df = self._process_noise_byseries(
                            1,
                            series_list,
                            chan,
                            chan_config,
                            nevents)
                        
                    else:
                        
                        # split data
                        series_list_split = self._split_series(
                            series_list, ncores)
                    
                        # lauch pool processing
                        if self._verbose:
                            print(f'INFO: Noise processing with be split '
                                  f'between {ncores} cores!')
            
                        node_nums = list(range(ncores+1))[1:]
                        pool = Pool(processes=ncores)
                        output_df_list = pool.starmap(
                            self._process_noise_byseries,
                            zip(node_nums,
                                series_list_split,
                                repeat(chan),
                                repeat(chan_config),
                                repeat(nevents)))
                        
                        pool.close()
                        pool.join()

                        # concatenate output
                        df_list = list()
                        for df in output_df_list:
                            if df is not None:
                                df_list.append(df)
                        if df_list:
                            output_df = pd.concat(df_list)

                    # add in filter_data
                    metadata = {'processing_id': self._processing_id,
                                'group_name': self._group_name}
                    
                    self._filter_data.set_noise_dataframe(chan, output_df,
                                                          metadata=metadata,
                                                          tag=filterdata_tag)
                    
                    # store in dictionary
                    if chan not in output_dict['noise'].keys():
                        output_dict['noise'][chan] = dict()
                    output_dict['noise'][chan][filterdata_tag] = {'df': output_df}
                    
                # calc  global noise 
                if chan_config['calc_noise_global']:
                    
                    data = self._process_noise_global(
                        series_list,
                        chan,
                        chan_config,
                        nevents,
                    )

                    if chan not in output_dict['noise']:
                        output_dict['noise'][chan] = dict()

                    for tag, tag_data in data.items():

                        pretrigger_samples = tag_data['pretrigger_length_samples']
                        pretrigger_msec = tag_data['pretrigger_length_msec']
                        sample_rate =  tag_data['sample_rate']
                                              
                        if tag not in output_dict['noise'][chan]:
                           output_dict['noise'][chan][tag] = dict()
                           
                        output_dict['noise'][chan][tag].update(tag_data.copy())

                        # add in filter_data
                        metadata = {
                            'processing_id': self._processing_id,
                            'group_name': self._group_name}
                        
                        # save in filter data
                        if 'psd' in tag_data:
                                                   
                            # two-sided 
                            self._filter_data.set_psd(
                                chan,
                                tag_data['psd'],
                                psd_freqs=tag_data['psd_freqs'],
                                sample_rate=sample_rate,
                                pretrigger_length_msec=pretrigger_msec,
                                pretrigger_length_samples=pretrigger_samples,
                                metadata=metadata,
                                tag=tag)
                            
                            # folded over
                            self._filter_data.set_psd(
                                chan,
                                tag_data['psd_fold'],
                                psd_freqs=tag_data['psd_freqs_fold'],
                                sample_rate=sample_rate,
                                pretrigger_length_msec=pretrigger_msec,
                                pretrigger_length_samples=pretrigger_samples,
                                metadata=metadata,
                                tag=tag)

                        if 'csd' in tag_data:
                            self._filter_data.set_csd(
                                chan,
                                tag_data['csd'],
                                csd_freqs=tag_data['csd_freqs'],
                                sample_rate=sample_rate,
                                pretrigger_length_msec=pretrigger_msec,
                                pretrigger_length_samples=pretrigger_samples,
                                metadata=metadata,
                                tag=tag)
                    
        # ================================
        # Template generation 
        # ================================
        
        if enable_template:                      
        
            # loop channel and process
            for chan in channels_template:
                chan_config = self._processing_config['template'][chan]
                data = self._process_template(chan, chan_config)
                
                for tag, tag_data in data.items():
                    self._filter_data.set_template(
                        chan,
                        tag_data['template'],
                        sample_rate=self._sample_rate,
                        pretrigger_length_msec=tag_data['pretrigger_length_msec'],
                        retrigger_length_samples=tag_data['pretrigger_length_samples'],
                        metadata=tag_data['template_metadata'],
                        tag=tag)
                
                    
        # ================================
        # SAVE data
        # ================================
        
        if lgc_save:
            
            # build file name
            file_name = save_file_path
            if (save_file_path is None
                or not os.path.isfile(save_file_path)):
                
                # save path
                if save_file_path is None:

                    save_file_path = self._base_path
                    save_file_path += '/filterdata'
                    if '/raw/filterdata' in save_file_path:
                        save_file_path = save_file_path.replace('/raw/filterdata',
                                                                '/filterdata')

                # add group name
                if self._group_name not in save_file_path:
                    save_file_path = save_file_path + '/' + self._group_name

                # create file name
                file_name = self._create_file_name(
                    save_file_path, self._facility, self._processing_id
                )
                
            # save 
            self._filter_data.save_hdf5(file_name, overwrite=True)
            
            
        if lgc_output:
            return output_dict
        
    def _process_didv(self, node_num,
                      series_list,
                      channel,
                      config,
                      nevents):
                     
        """
        Process dIdV data
        
        Parameters
        ---------

        node_num :  int
          node id number, used for display
        
        series_list : str
          list of series name to be processed
        
        channel : str
          channel name
        
        nevents : int (or None)
          maximum number of events

        """

        # node string (for display)
        node_num_str = str()
        if node_num>-1:
            node_num_str = 'Node #' + str(node_num)

        # IV sweep results
        file_name = None
        iv_type = 'noise'
        results = None
        
        if 'ivsweep_file' in config:
            file_name = config['ivsweep_file']
            if 'ivsweep_result_type' in  config:
               iv_type =  config['ivsweep_result_type']
        elif 'ivsweep_data' in config:
            results = config['ivsweep_data']
        else:
            raise ValueError(
                f'ERROR: IVsweep results required for channel '
                f'{channel}, either "ivsweep_file"   or '
                f'"ivsweep_data" (dictionary)')


        # other flags
        bias_parameter_type = config['bias_parameter_type']
        inf_loop_gain_approx  = config['auto_infinite_lgain']

        calc_true_current = False
        if bias_parameter_type == 'true_current':
            calc_true_current = True
        elif  bias_parameter_type == 'infinite_lgain_current':
            inf_loop_gain_approx = True
                 
        # initialize data dict
        ivsweep_params = ['group_name_sweep',
                          'i0', 'i0_err', 'r0', 'r0_err',
                          'p0', 'p0_err',
                          'ibias', 'ibias_err']
        
        bias_params = ['rp', 'rn', 'rshunt',
                       'i0','i0_err',
                       'r0','r0_err',
                       'p0', 'p0_err',
                       'ibias', 'biasparams_type']
        
        smallsignal_params = ['l','L', 'tau0', 'beta', 'gratio']
        fit_params = ['didv0', 'chi2', 'tau+','tau-','tau3']
       
        config_params = ['tes_bias', 'temperature_mc', 'temperature_cp',
                         'temperature_still']

        infinite_lgain_params = ['i0','r0', 'p0']
                     
        output_dict = {'series_name': list(),
                       'group_name': list(),
                       'tmean': list(),
                       'didvmean':list(),
                       'didvstd': list(),
                       'offset_didv': list(),
                       'offset_err_didv': list(),
                       'fs' : list(),
                       'sgfreq': list(),
                       'sgamp':list()}
        
        for param in ivsweep_params:
            param += '_ivsweep'
            output_dict[param] = list()
            
        for param in bias_params:
            output_dict[param] = list()
        
        for param in  smallsignal_params:
            param_val = param + '_3poles_fit'
            param_err = param + '_err_3poles_fit'
            output_dict[param_val] = list()
            output_dict[param_err] = list()

        for param in fit_params:
            param += '_3poles_fit'
            output_dict[param] = list()

        for param in config_params:
            output_dict[param] = list()
       
        for param in infinite_lgain_params:
            param += '_infinite_lgain'
            output_dict[param] = list()

        output_dict['time_since_group_start'] = list()
        output_dict['time_since_fridge_run_start'] = list()
        output_dict['series_time'] = list()
            
        # loop series
        for series in series_list:

            if self._verbose:
                print(f'INFO {node_num_str}: Processing dIdV '
                      f'series {series}!') 
            
            # get files and config
            file_list = self._raw_data_exttrig[series]
            if channel not in self._det_config_exttrig[series].keys():
                continue
            
            detconfig =  self._det_config_exttrig[series][channel]
            seriesconfig = self._series_config_exttrig[series]
            
            # check if dIdV
            is_didv = (detconfig['signal_gen_onoff']=='on'
                       and detconfig['signal_gen_source']=='tes')

            if not is_didv:
                continue

            # Instantiate 
            didvanalysis = DIDVAnalysis(verbose=True)

            # Process data
            didvanalysis.process_raw_data(channel, file_list, series=series)

            # Set IV Sweep
            if file_name is not None:
                didvanalysis.set_ivsweep_results_from_file(
                    channel,
                    file_name,
                    iv_type=iv_type,
                    include_bias_parameters=True)

            elif results is not None:
                didvanalysis.set_ivsweep_results_from_data(
                    channel,
                    results,
                    iv_type=iv_type
                )
                
            else:
                raise ValueError(f'ERROR: No IV sweep avalaible for '
                                 f'channel {channel}')
                
            # Fit 2+3 poles
            didvanalysis.dofit([2,3], lgc_plot=False)

            # Calc small signal parameters
            didvanalysis.calc_smallsignal_params(
                calc_true_current=calc_true_current,
                inf_loop_gain_approx=inf_loop_gain_approx
            )
            
            # Calc infinite loop gain parameters
            didvanalysis.calc_bias_params_infinite_loop_gain()
            
            # Store in dataframe
            didv_data  = didvanalysis.get_didv_data(channel)
            fitresults = didvanalysis.get_fit_results(channel, 3)
            
            # Fill data
            output_dict['series_name'].append(series)
            output_dict['group_name'].append(self._group_name)
            output_dict['tmean'].append(didv_data['didvobj']._tmean) 
            output_dict['didvmean'].append(didv_data['didvobj']._didvmean) 
            output_dict['didvstd'].append(didv_data['didvobj']._didvstd) 
            output_dict['offset_didv'].append(didv_data['didvobj']._offset)
            output_dict['offset_err_didv'].append(didv_data['didvobj']._offset_err)
            output_dict['fs'].append(didv_data['didvobj']._fs) 
            output_dict['sgfreq'].append(didv_data['didvobj']._sgfreq) 
            output_dict['sgamp'].append(didv_data['didvobj']._sgamp) 

            # IV sweep
            for param in ivsweep_params:
                param_val = np.nan
                if ('ivsweep_results' in didv_data
                    and param in didv_data['ivsweep_results']):
                    param_val =  didv_data['ivsweep_results'][param]
                    
                output_dict[param + '_ivsweep'].append(param_val) 

            # bias params
            for param in bias_params:
                param_val = fitresults['biasparams'][param]
                output_dict[param].append(param_val) 
            
            # 3-poles SSP
            for param in  smallsignal_params:
                param_val = fitresults['ssp_light']['vals'][param]
                param_val_err = fitresults['ssp_light']['sigmas']['sigma_' + param]
                output_dict[param + '_3poles_fit'].append(param_val)
                output_dict[param + '_err_3poles_fit'].append(param_val_err)

            # fit params
            output_dict['didv0_3poles_fit'] = fitresults['didv0']
            output_dict['chi2_3poles_fit'] = fitresults['cost']
            output_dict['tau+_3poles_fit'] = fitresults['falltimes'][0]
            output_dict['tau-_3poles_fit'] = fitresults['falltimes'][1]
            output_dict['tau3_3poles_fit'] = fitresults['falltimes'][2]

            for param in config_params:
                param_val = np.nan
                if param in didv_data['data_config']:
                    param_val = didv_data['data_config'][param]
                output_dict[param].append(param_val)
           
            for param in infinite_lgain_params:
                param_val = fitresults['biasparams_infinite_lgain'][param]
                param += '_infinite_lgain'
                output_dict[param].append(param_val)

            series_time = int(seriesconfig['timestamp'])
            output_dict['series_time'].append(series_time)
            
            series_start = np.nan
            if 'group_start' in seriesconfig:
                series_start = series_time - int(seriesconfig['group_start'])
            output_dict['time_since_group_start'].append(series_start)

            
            run_start = np.nan
            if 'fridge_run_start' in seriesconfig:
                run_start = series_time - int(
                    seriesconfig['fridge_run_start']
                )
            output_dict['time_since_fridge_run_start'].append(run_start)

        # convert to dataframe
        df = pd.DataFrame.from_dict(output_dict)
        return df


    def _process_noise_byseries(self, node_num,
                                series_list,
                                channel,
                                channel_config,
                                nevents):
        """
        Process noise by series
        
        Parameters
        ---------

        node_num :  int
          node id number, used for display
        
        series_list : str
          list of series name to be processed
        
        channel : str
          channel name
        
        nevents : int (or None)
          maximum number of events

        """

        # node string (for display)
        node_num_str = str()
        if node_num>-1:
            node_num_str = 'Node #' + str(node_num)
            
        # check if single channel or multi channel
        is_multi_channels = len(channel.split('|')) > 1

        # initialie data dict
        output_dict = {
            'group_name':list(),
            'series_name': list(),
            'time_since_group_start': list(),
            'time_since_fridge_run_start':list(),
            'series_time': list()
        }
        
        config_params = ['temperature_mc',
                         'temperature_cp',
                         'temperature_still']  

        for param in config_params:
            output_dict[param] = list()
    
        
        if is_multi_channels:
             output_dict.update(
                 {'corrcoeff_fold': list(),
                  'corrcoeff_freqs_fold': list()}
             )
        else:
            output_dict.update(
                {'psd': list(),
                 'psd_freqs': list(),
                 'psd_fold': list(),
                 'psd_freqs_fold': list(),
                 'offset': list()}
            )
            
        # configuration
        filterdata_tag = channel_config['byseries_noise_tag']
        psd_amp_freq_ranges = channel_config['psd_amp_freq_range_list']
        corr_coeff_freq_ranges = channel_config['corr_coeff_freq_range_list']

        # trace length (tag dependent)
        tag_config =  channel_config[filterdata_tag]
        trace_samples = tag_config['trace_length_samples']
        trace_msec = tag_config['trace_length_msec']
        pretrigger_samples = tag_config['pretrigger_length_samples']
        pretrigger_msec = tag_config['pretrigger_length_msec']
            
        # loop series
        for series in series_list:

            if self._verbose:
                print(f'INFO {node_num_str}: Processing noise for '
                      f'channel {channel}, series {series}!') 
            
            # get files and config
            file_list = self._raw_data_notrig[series]

            # series config
            seriesconfig = self._series_config_notrig[series]

            # det config (take one channel if multiple)
            channels_split, sep = utils.split_channel_name(
                channel, self._channels_notrig)
            if channels_split[0] not in self._det_config_notrig[series].keys():
                raise ValueError(f'ERROR: channel {channels_split[0]} not '
                                 f'in raw data')
            detconfig =  self._det_config_notrig[series][channels_split[0]]
          
            # output dict
            output_dict['series_name'].append(series)
            output_dict['group_name'].append(self._group_name)
        
            # for continuous data -> generate randoms
            # for randoms data -> set raw data
            
            # instantiate noise
            noise_inst = Noise(verbose=False)

            if self._is_continuous:
                noise_inst.generate_randoms(file_list, nevents=nevents,
                                            min_separation_msec=100,
                                            ncores=1)
            else:
                noise_inst.set_randoms(file_list)
                
            if not is_multi_channels:

                # calculate noise psd
                noise_inst.calc_psd(
                    channel,
                    trace_length_msec=trace_msec,
                    pretrigger_length_msec= pretrigger_msec,
                    trace_length_samples=trace_samples,
                    pretrigger_length_samples=pretrigger_samples,
                    nevents=nevents)

                # get psd
                psd, psd_freqs = noise_inst.get_psd(channel)
                psd_fold, psd_freqs_fold = noise_inst.get_psd(channel, fold=True)
                
                # get offset
                offset = noise_inst.get_offset(channel)
            
                # psd ranges
                if psd_amp_freq_ranges is not None:

                    # intialize list of parameters
                    psd_amp_names, psd_amp_ind_ranges = self._get_ranges(
                        psd_freqs_fold, psd_amp_freq_ranges
                    )

                    
                    for it, ind_range in enumerate(psd_amp_ind_ranges):
                        
                        ind_low = ind_range[0]
                        ind_high = ind_range[1]
                                             
                        # take median
                        psd_avg = np.average(psd_fold[ind_low:ind_high])
                        psd_avg = np.sqrt(psd_avg)
                    
                        # parameter name
                        psd_amp_name = 'psd_amp_' + psd_amp_names[it]
                        if psd_amp_name not in output_dict.keys():
                            output_dict[psd_amp_name] = list()

                        # add
                        output_dict[psd_amp_name].append(psd_avg)
                        
                    # fill dictionary
                    output_dict['psd'].append(psd)
                    output_dict['psd_freqs'].append(psd_freqs)
                    output_dict['psd_fold'].append(psd_fold)
                    output_dict['psd_freqs_fold'].append(psd_freqs_fold)
                    output_dict['offset'].append(offset)
            else:

                channel_names = channel.split('|')
                
                # calculate noise csd
                noise_inst.calc_csd(
                    channel,
                    trace_length_msec=trace_msec,
                    pretrigger_length_msec= pretrigger_msec,
                    trace_length_samples=trace_samples,
                    pretrigger_length_samples=pretrigger_samples,
                    nevents=nevents)
             
                coeff, coeff_freqs = noise_inst.get_corrcoeff(
                    channel,
                    tag=filterdata_tag,
                    fold=True)
                              
                if corr_coeff_freq_ranges is not None:
                    
                    for ii in range(coeff.shape[0]):
                        for jj in range(coeff.shape[1]):

                            if ii <= jj:
                                continue
                            
                            chan_label = f'{channel_names[ii]}-{channel_names[jj]}'
                            print(chan_label)
                            
                            # intialize list of parameters
                            corr_coeff_names, corr_coeff_ind_ranges = self._get_ranges(
                                coeff_freqs, corr_coeff_freq_ranges
                            )
                            
                            coeff_array = savgol_filter(
                                coeff[ii][jj][1:], 7, 3,
                                mode='nearest')
                            
                            for it, ind_range in enumerate(corr_coeff_ind_ranges):
                        
                                ind_low = ind_range[0]
                                ind_high = ind_range[1]
                                                     
                                # take average
                                corr_coeff_avg = np.average(coeff_array[ind_low:ind_high])

                                # parameter name
                                corr_coeff_name = ('corrcoeff_'
                                                   + chan_label
                                                   + '_' + corr_coeff_names[it])
                                
                                if corr_coeff_name not in output_dict.keys():
                                    output_dict[corr_coeff_name] = list()

                                # add
                                output_dict[corr_coeff_name].append(corr_coeff_avg)
                        
                # fill dictionary
                output_dict['corrcoeff_fold'].append(coeff)
                output_dict['corrcoeff_freqs_fold'].append(coeff_freqs)
            
            # config
            for param in config_params:
                param_val = np.nan
                if param in detconfig:
                    param_val = detconfig[param]
                output_dict[param].append(param_val)
           
            # timing
            series_time = int(seriesconfig['timestamp'])
            output_dict['series_time'].append(series_time)
            
            series_start = np.nan
            if 'group_start' in seriesconfig:
                series_start = series_time - int(seriesconfig['group_start'])
            output_dict['time_since_group_start'].append(series_start)
            
            run_start = np.nan
            if 'fridge_run_start' in seriesconfig:
                run_start = series_time - int(
                    seriesconfig['fridge_run_start']
                )
            output_dict['time_since_fridge_run_start'].append(run_start)
            
        # convert to dataframe
        df = pd.DataFrame.from_dict(output_dict)
        return df


    
    def _process_noise_global(self, series_list,
                              channel,
                              channel_config,
                              nevents):
        
        """
        Process global noise psd
        
        Parameters
        ---------
        
        series_list : str
          list of series name to be processed
        
        channel : str
          channel name
        
        nevents : int (or None)
          maximum number of events

        """

        if self._verbose:
            print(f'INFO: Processing global noise psd')

        # initialize output
        output_dict = dict()

            
        # instantiate noise
        noise_inst = Noise(verbose=True)

        # generate randoms
        data_path = self._base_path + '/' + self._group_name
        
        if self._is_continuous:
            noise_inst.generate_randoms(data_path,
                                        series=series_list,
                                        nevents=nevents,
                                        min_separation_msec=100,
                                        ncores=1)
        else:
            noise_inst.set_randoms(self._base_path,
                                   series=series_list)
                  
        # loop tags
        noise_tags = channel_config['global_noise_tag_list']
        
        for tag in noise_tags:

            tag_config = channel_config[tag]
            trace_samples = tag_config['trace_length_samples']
            trace_msec = tag_config['trace_length_msec']
            pretrigger_samples = tag_config['pretrigger_length_samples']
            pretrigger_msec = tag_config['pretrigger_length_msec']
                        
            # nb_channels
            nb_channels = len(channel.split('|'))
          
            if nb_channels == 1:
                
                noise_inst.calc_psd(
                    channel,
                    trace_length_msec=trace_msec,
                    pretrigger_length_msec= pretrigger_msec,
                    trace_length_samples=trace_samples,
                    pretrigger_length_samples=pretrigger_samples,
                    nevents=nevents)

                psd, psd_freqs = noise_inst.get_psd(channel)
                psd_fold, psd_freqs_fold = noise_inst.get_psd(channel, fold=True)
                sample_rate = noise_inst.get_sample_rate()
                              
                output_dict[tag] = {
                    'group_name': self._group_name,
                    'psd':psd,
                    'psd_freqs':psd_freqs,
                    'psd_fold':psd_fold,
                    'psd_freqs_fold':psd_freqs_fold,
                    'pretrigger_length_msec': pretrigger_msec,
                    'pretrigger_length_samples': pretrigger_samples,
                    'sample_rate': sample_rate
                }

                
            else:

                noise_inst.calc_csd(
                    channel,
                    trace_length_msec=trace_msec,
                    pretrigger_length_msec=pretrigger_msec,
                    trace_length_samples=trace_samples,
                    pretrigger_length_samples=pretrigger_samples,
                    nevents=nevents)
                

                csd, csd_freqs = noise_inst.get_csd(channel)
                sample_rate = noise_inst.get_sample_rate()
                
                output_dict[tag] = {
                    'group_name': self._group_name,
                    'csd': csd,
                    'csd_freqs':csd_freqs,
                    'pretrigger_length_msec': pretrigger_msec,
                    'pretrigger_length_samples': pretrigger_samples,
                    'sample_rate': sample_rate
                }
                
        return output_dict

    def _process_template(self, 
                          channel,
                          channel_config):
        
        """
        Create templates
        
        Parameters
        ---------
                
        channel : str
          channel name
        
        channel_config: dict
           user configuration data (read from yaml file)

        """

        if self._verbose:
            print(f'INFO: Creating templates for channel {channel}')

        # initialize output
        output_dict = dict()

        # Instantiate template
        template_inst = Template(verbose=True)

        # loop tags
        for tag in channel_config.keys():

            # output
            if tag not in  output_dict:
                 output_dict[tag] = dict()

            tag_config = channel_config[tag]

            # poles 
            template_poles = int(tag_config['template_poles'])

            # trace/pretrigger length
            trace_length_msec = None
            pretrigger_length_msec = None
            trace_length_samples = None
            pretrigger_length_samples = None

            if 'trace_length_msec' in tag_config:
                trace_length_msec = tag_config['trace_length_msec']
            if 'trace_length_samples' in tag_config:
                trace_length_samples = tag_config['trace_length_samples']
            if 'pretrigger_length_msec' in tag_config:
                pretrigger_length_msec = tag_config['pretrigger_length_msec']
            if 'pretrigger_length_samples' in tag_config:
                pretrigger_length_samples = tag_config['pretrigger_length_samples']
            

            # mandatory parameters
            A = tag_config['amplitude_A']
            rise_time = tag_config['rise_time']
            fall_time_1 = tag_config['fall_time_1']

            if not isinstance(A, list):
                A = [A]
            if not isinstance(rise_time, list):
                rise_time = [rise_time]
            if not isinstance(fall_time_1, list):
                fall_time_1 = [fall_time_1]

            nb_functions = len(A)
            
                
            # other amplitudes
            B = [None]*nb_functions
            C = [None]*nb_functions
        
            if 'amplitude_B' in tag_config:
                B = tag_config['amplitude_B']
                if not isinstance(B, list):
                    B = [B]
            if 'amplitude_C' in tag_config:
                C = tag_config['amplitude_C']
                if not isinstance(C, list):
                    C = [C]
         
            # fall times
            fall_time_2 = [None]*nb_functions
            fall_time_3 = [None]*nb_functions
                                
            if 'fall_time_2' in tag_config:
                fall_time_2 = tag_config['fall_time_2']
                if not isinstance(fall_time_2, list):
                    fall_time_2 = [fall_time_2]
                    
            if 'fall_time_3' in tag_config:
                fall_time_3 = tag_config['fall_time_3']
                if not isinstance(fall_time_3, list):
                    fall_time_3 = [fall_time_3]
            
            # 2-poles
            if  template_poles == 2:

                if len(A) == 1:
                    template.create_template(
                        channel,
                        sample_rate=self._sample_rate,
                        trace_length_msec=trace_length_msec,
                        trace_length_samples=trace_length_samples,
                        pretrigger_length_msec=pretrigger_length_msec,
                        pretrigger_length_samples=pretrigger_length_samples,
                        A=A[0],
                        tau_r=rise_time[0],
                        tau_f1=fall_time_1[0],
                        tag=tag)
                else:
                    template.create_template_sum_twopoles(
                        channel,
                        A,
                        rise_time,
                        fall_time_1,
                        sample_rate=self._sample_rate,
                        trace_length_msec=trace_length_msec,
                        trace_length_samples=trace_length_samples,
                        pretrigger_length_msec=pretrigger_length_msec,
                        pretrigger_length_samples=pretrigger_length_samples,
                        tag=tag)
            else:

                template.create_template(
                    channel,
                    sample_rate=self._sample_rate,
                    trace_length_msec=trace_length_msec,
                    trace_length_samples=trace_length_samples,
                    pretrigger_length_msec=pretrigger_length_msec,
                    pretrigger_length_samples=pretrigger_length_samples,
                    A=A[0],
                    B=B[0],
                    C=C[0],
                    tau_r=rise_time[0],
                    tau_f1=fall_time_1[0],
                    tau_f2=fall_time_1[0],
                    tau_f3=fall_time_1[0],
                    tag=tag)
                    
            # get template
            output_dict[tag]['template'] = template_inst.get_template(
                channel,
                tag=tag)
            output_dict[tag]['template_metadata'] = copy.deepcopy(tag_config)

        return output_dict
                
            
    def _get_file_list(self, files_or_path, series=None,
                       restricted=False):
        """
        Get file list from path. Return as a dictionary
        with key=series and value=list of files

        Parameters
        ----------
        
        files_or_path : str or list of str 
           raw data group directory OR full path to HDF5  file 
           (or list of files). Only a single raw data group 
           allowed 
        
        series : str or list of str, optional
            series to be process, disregard other data from raw_path
        
        restricted : boolean
            if True, use restricted data
            if False, exclude restricted data



        Return
        -------
        
        series_dict : dict 
          list of files for splitted inot series

        base_path :  str
           base path of the raw data

        group_name : str
           group name of raw data

        """
        
        # convert input to list if needed
        if not isinstance(files_or_path, list):
            files_or_path = [files_or_path]
        
        # initialize
        file_list = list()
        base_paths = list()
        group_names = list()

        # 1. Get list of all files
        for a_path in files_or_path:
            
            # case path is a directory
            if os.path.isdir(a_path):

                # append to list 
                base_paths.append(str(Path(a_path).parent))
                group_names.append(str(Path(a_path).name))
                
                if series is not None:
                    if series == 'even' or series == 'odd':
                        file_name_wildcard = series + '_*.hdf5'
                        file_list.extend(glob(a_path + '/' + file_name_wildcard))
                    else:
                        if not isinstance(series, list):
                            series = [series]
                        for a_series in series:
                            # convert to series name?
                            file_name_wildcard = '*' + a_series + '_*.hdf5'
                            file_list.extend(glob(a_path + '/' + file_name_wildcard))
                else:
                    file_list.extend(glob(a_path + '/*.hdf5'))
                                    
                    
            # case file
            elif os.path.isfile(a_path):

                base_paths.append(str(Path(a_path).parents[1]))
                group_names.append(str(Path(Path(a_path).parent).name))
                    
                if a_path.find('.hdf5') != -1:
                    
                    if series is not None:
                        if series == 'even' or series == 'odd':
                            if a_path.find(series) != -1:
                                file_list.append(a_path)
                        else:
                            if not isinstance(series, list):
                                series = [series]
                            for it_series in series:
                                if a_path.find(it_series) != -1:
                                    file_list.append(a_path)
                    else:
                        file_list.append(a_path)

            else:
                raise ValueError('File or directory "'
                                 + a_path
                                 + '" does not exist!')
            
        if not file_list:
            raise ValueError('ERROR: No raw input data found. Check arguments!')

        # base path
        base_paths = list(set(base_paths))
        if len(base_paths) != 1:
            raise ValueError('ERROR: Found raw data from multiple directories. '
                             'Only a single base directory allowed!')

        base_path = base_paths[0]

        # group name
        group_names = list(set(group_names))
        if len(group_names) != 1:
            raise ValueError('ERROR: Found raw data with multiple group name. '
                             'Only a single group allowed!')

        group_name = group_names[0]
        
        # make uniqye sort
        file_list = list(set(file_list))
        file_list.sort()


        # initialize reader
        h5reader = h5io.H5Reader()
        trigger_types = ['rand', 'cont', 'exttrig',
                         'threshtrig']

        # Separate into 3  types of files
        # and store in series dictionary
        
        #  1. exttrig ("exttrig" or "didv")
        #  2. notrig  ("rand" or "cont")
        #  3. threshtrig ("threshold")
        #

        series_exttrig_dict = dict()
        series_notrig_dict = dict()
        series_threshtrig_dict = dict()
        file_counter_exttrig = 0
        file_counter_notrig = 0
        file_counter_threshtrig = 0
        series_name = None
            
        for afile in file_list:

            # file name
            file_name = str(Path(afile).name)
            
            # skip if filter file (with name "filter")
            if 'filter' in file_name:
                continue

            # restricted
            if (restricted
                and 'restricted' not in file_name):
                continue

            # not restricted
            if (not restricted
                and 'restricted' in file_name):
                continue

            # check file type
            file_type = None
            if ('rand_' in  file_name
                or 'cont_'  in  file_name):
                file_type = 'notrig'
                            
            elif ('didv_' in file_name
                  or 'exttrig_' in file_name):
                file_type = 'exttrig'
            elif 'thresh' in file_name:
                file_type = 'threshtrig'
                

            if file_type is None:
                # check trigger type of first event
                metadata = h5reader.get_metadata(afile)
                data_mode = None
                if 'adc_list' in metadata:
                    adc_name = metadata['adc_list'][0]
                    data_mode = metadata['groups'][adc_name]['data_mode']
             
                if data_mode is not None:
                    if data_mode == 'cont':
                        file_type = 'notrig'
                    elif data_mode == 'trig-ext':
                        file_type = 'exttrig'
                    elif data_mode == 'threshold':
                        file_type = 'threshtrig'
                    
            if file_type is None:
                print(f'WARNING: file {file_name} not recognized! '
                      f'Skipping...')
                continue

            # get series name
            if (series_name is None
                or  series_name not in file_name):
                file_info = h5reader.get_file_info(afile)
                series_name = str(
                    h5io.extract_series_name(
                        file_info['series_num']
                    )
                )

            # split data type
            if  file_type == 'notrig':
                
                if series_name not in series_notrig_dict.keys():
                    series_notrig_dict[series_name] = list()
                    
                if afile not in  series_notrig_dict[series_name]:
                    series_notrig_dict[series_name].append(afile)
                    file_counter_notrig += 1
            
            elif  file_type == 'exttrig':
                
                if series_name not in series_exttrig_dict.keys():
                    series_exttrig_dict[series_name] = list()
                    
                if afile not in series_exttrig_dict[series_name]:
                    series_exttrig_dict[series_name].append(afile)
                    file_counter_exttrig += 1
                            
            elif  file_type == 'threshtrig':
                
                if series_name not in series_threshtrig_dict.keys():
                    series_threshtrig_dict[series_name] = list()
                    
                if afile not in series_threshtrig_dict[series_name]:
                    series_threshtrig_dict[series_name].append(afile)
                    file_counter_threshtrig += 1
                    
                
        if self._verbose:

            if file_counter_notrig > 0:
                print(f'INFO: Found total of {file_counter_notrig} '
                      f'continuous/randoms files from '
                      f'{len(series_notrig_dict.keys())} '
                      f'different series number!')

            if file_counter_exttrig > 0:
                print(f'INFO: Found total of {file_counter_exttrig} '
                      f'didv/external trigger files from '
                      f'{len(series_exttrig_dict.keys())} '
                      f'different series number!')
                
            if file_counter_threshtrig > 0:
                print(f'INFO: Found total of {file_counter_threshtrig} '
                      f'threshold trigger files from '
                      f'{len(series_threshtrig_dict.keys())} '
                      f'different series number!')
                
                
      
        return (series_notrig_dict, series_exttrig_dict, 
                series_threshtrig_dict, base_path, group_name)

    
    def _get_data_config(self, raw_data_dict):
        """
        Get raw data config for each series
        """

        config = dict()
        channels = list()
        info = dict()
        h5reader = h5io.H5Reader()
        for series, file_list in raw_data_dict.items():
            file_name = file_list[0]
            config[series] = h5reader.get_detector_config(file_name)
            info[series] = h5reader.get_file_info(file_name)
                    
            for chan in config[series].keys():
                if chan not in channels:       
                    channels.append(chan)

    
        return info, config, channels
        

    def _is_continuous(self, raw_data_dict):
        """
        Check if data continuous or randoms
        """

        is_continuous = False
        for series, file_list in raw_data_dict.items():
            if not file_list:
                continue
            file_name = file_list[0]
            name = str(Path(file_name).name)
            if 'cont_' in name:
                is_continuous = True
            break

        return  is_continuous

    
    def _create_file_name(self, save_file_path, facility, processing_id):
        
        """
        Create output directory 

        Parameters
        ----------
        
        save_file_path :  str
           full path to base directory 
        
        facility : int
           id #  of facility 
        
        processing_id : str
           processing id 
        
    
        Return
        ------
          file_name : str
            full path to file

        """

        now = datetime.now()
        series_day = now.strftime('%Y') +  now.strftime('%m') + now.strftime('%d') 
        series_time = now.strftime('%H') + now.strftime('%M')
        series_name = ('I' + str(facility) +'_D' + series_day + '_T'
                       + series_time + now.strftime('%S'))
        
                
        # prefix
        prefix = 'filter'
        if processing_id is not None:
            prefix = processing_id + '_filter'

        # create directory if needed
        if not os.path.isdir(save_file_path):
            try:
                os.makedirs(save_file_path)
                os.chmod(output_dir,
                         stat.S_IRWXG | stat.S_IRWXU
                         | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'
                                 + save_file_path  + '"!\n')

        # build file name
        file_name = save_file_path + '/' + prefix + '_' + series_name + '.hdf5'
                
        return file_name
        

    def _split_series(self, series_list, ncores):
        """
        Split data  between nodes
        following series


        Parameters
        ----------

        ncores : int
          number of cores

        Return
        ------

        output_list : list
           list of dictionaries (length=ncores) containing 
           data
         

        """

        output_list = list()
        
        # split series
        series_split = np.array_split(series_list, ncores)

        # remove empty array
        for series_sublist in series_split:
            if series_sublist.size == 0:
                continue
            output_list.append(list(series_sublist))
            

        return output_list


    def _read_config(self, yaml_file, available_channels):
        """
        Read config file
        """

        
        config = utils.read_config(yaml_file, available_channels)

        if ('didv' not in config
            and 'noise' not in config
            and 'template' not in config):
            raise ValueError(f'ERROR: No processing configuration found '
                             f'in {yaml_file}!')

        return config



    def __ivsweep_data(self, channels,
                       file_name=None,
                       results=None,
                       calc_true_current=False,
                       iv_type='noise'):
        """
        check IV Sweep data 
            - either stored in hdf5 file
            - or dictionary/pandas series
        """

        # check input
        if (file_name is not None and results is not None): 
            raise ValueError('ERROR: Choose between "file_name" '
                             'or "data" dictionary!')
        
        # channels
        if isinstance(channels, str):
            channels = [channels]

        # initialize if needed
        output_dict = dict()
        
        for chan in channels:
            output_dict[chan] = {
                'file': None,
                'iv_type': iv_type,
                'results': None
            }
            
        # check file name
        if file_name is not None:
            
            if not os.path.isfile(file_name):
                raise ValueError(f'ERROR:  {file_name} not found!')

            for chan in channels:
                output_dict[chan]['file'] = file_name
                output_dict[chan]['iv_type'] = iv_type
                         
        elif results is not None:

            if isinstance(results, pd.Series):
                results = results.to_dict()
            elif not isinstance(results, dict):
                raise ValueError(
                    'ERROR: "results" should be a dictionary '
                    ' or pandas series')

            # check some mandatory parameters
            if 'rp' not in results.keys():
                raise ValueError('ERROR: Missing "rp" in results dictionary!')
            
            if ('rsh' not in results.keys()
                and 'rshunt' not in results.keys()):
                raise ValueError('ERROR: Missing "rshunt" in results dictionary!')
            
            for chan in channels:
                output_dict[chan][chan]['iv_type'] = iv_type
                output_dict[chan][chan]['results'] = results.copy()
                
    def _get_ranges(self, freqs, freq_ranges):
        """
        convert frequency ranges to index ranges
        """

        name_list = list()
        index_ranges = list()
        
        for it, freq_range in enumerate(freq_ranges):
                        
            # ignore if not a range
            if len(freq_range) != 2:
                continue
            
            # low/high freqeucy
            f_low = abs(freq_range[0])
            f_high = abs(freq_range[1])
                    
            if f_low > f_high:
                f_high = f_low + (f_low - f_high)
                        
            # indices
            ind_low = np.argmin(np.abs(freqs - f_low))
            ind_high = np.argmin(np.abs(freqs - f_high))
            if (ind_low == ind_high):
                if ind_low < len(freqs)-2:
                    ind_high = ind_low + 1
                else:
                    continue
                
            # store
            name = (str(round(f_low))
                         + '_'
                         + str(round(f_high)))

            if name in name_list:
                continue
            
            name_list.append(name)
            index_ranges.append((ind_low, ind_high))

            
        return name_list, index_ranges
