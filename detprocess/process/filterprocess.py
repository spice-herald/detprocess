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
from humanfriendly import parse_size
from itertools import groupby
from detprocess.core import Noise, DIDVAnalysis, FilterData, Template, NoiseModel
from detprocess import utils
import pytesdaq.io as h5io


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
                    
        # iv sweep results (hdf5 file name or data dictionary)
        self._ivsweep_results = dict()

        # auto infinite loop gain
        self._auto_infinite_lgain = dict()
        
        # dIdV bias params type
        self._calc_true_current =  dict()
             
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

    
    def set_ivsweep_results(self, file_name=None,
                            results=None,
                            iv_type='noise',
                            channels=None):
        """
        Set IV Sweep data 
            - either stored in hdf5 file
            - or dictionary/pandas series
        """

        # check input
        if (file_name is not None and results is not None): 
            raise ValueError('ERROR: Choose between "file_name" '
                             'or "data" dictionary!')
        
        # channels
        if channels is None:
            channels = self._channels_exttrig
        elif isinstance(channels, str):
            channels = [channels]

        # initialize if needed
        for chan in channels:
            if chan not in self._ivsweep_results:
                self._ivsweep_results[chan] = {
                    'file': None,
                    'iv_type': iv_type,
                    'results': None
                }
            
        # check file name
        if file_name is not None:
            
            if not os.path.isfile(file_name):
                raise ValueError(f'ERROR:  {file_name} not found!')

            for chan in channels:
                self._ivsweep_results[chan]['file'] = file_name
                self._ivsweep_results[chan]['iv_type'] = iv_type
                         
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
                self._ivsweep_results[chan]['iv_type'] = iv_type
                self._ivsweep_results[chan]['results'] = results.copy()
                
  
    def set_auto_infinite_lgain(self, do_auto_infinite_lgain, channels=None):
        """
        Set type of bias paramerters I0, R0, either from IV sweep, or 
        calculated with true current, or infinite loop gain approximation
        """

        if not isinstance(do_auto_infinite_lgain, bool):
            raise ValueError('ERROR: expecting boolean input!')
        
        # channels
        if channels is None:
            channels = self._channels_exttrig
        elif isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            self._auto_infinite_lgain[chan] = do_auto_infinite_lgain


    def set_calc_true_current(self, do_calc, channels=None):
        """
        Set type of bias paramerters I0, R0, either from IV sweep, or 
        calculated with true current, or infinite loop gain approximation
        """

        if not isinstance(do_calc, bool):
            raise ValueError('ERROR: expecting boolean input!')
        
        # channels
        if channels is None:
            channels = self._channels_exttrig
        elif isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            self._calc_true_current[chan] = do_calc

    def proces_didv(self,
                    channels=None,
                    processing_id=None,
                    lgc_output=False,
                    lgc_save=False,
                    save_file_path=None,
                    filterdata_tag='default',
                    ncores=1):
        """
        Processing dIdV
        """

        self.process(channels=channels,
                     enable_didv=True,
                     processing_id=processing_id,
                     lgc_output=lgc_output,
                     lgc_save=lgc_save,
                     save_file_path=save_file_path,
                     filterdata_tag=filterdata_tag,
                     ncores=ncores)


    def proces_noise(self,
                     channels=None,
                     calc_psd_byseries=None,
                     calc_psd_global=None,
                     trace_length_msec=None,
                     pretrigger_length_msec=None,
                     trace_length_samples=None,
                     pretrigger_length_samples=None,
                     processing_id=None,
                     lgc_output=False,
                     lgc_save=False,
                     save_file_path=None,
                     filterdata_tag='default',
                     ncores=1):
        """
        Processing Noise
        """

        self.process(channels=channels,
                     enable_noise=True,
                     calc_psd_byseries=calc_psd_byseries,
                     calc_psd_global=calc_psd_global,
                     trace_length_msec=trace_length_msec,
                     pretrigger_length_msec=pretrigger_length_msec,
                     trace_length_samples=trace_length_samples,
                     pretrigger_length_samples=pretrigger_length_samples,
                     processing_id=processing_id,
                     lgc_output=lgc_output,
                     lgc_save=lgc_save,
                     save_file_path=save_file_path,
                     filterdata_tag=filterdata_tag,
                     ncores=ncores)

            
    def process(self,
                channels=None,
                enable_noise=False,
                enable_didv=False,
                enable_template=False,
                calc_psd_byseries=True,
                calc_psd_global=False,
                trace_length_msec=None,
                pretrigger_length_msec=None,
                trace_length_samples=None,
                pretrigger_length_samples=None,
                psd_amp_freq_ranges=None,
                nevents=None,
                processing_id=None,
                lgc_output=False,
                lgc_save=False,
                save_file_path=None,
                filterdata_tag='default',
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

        # check processing
        if enable_didv:

            if not self._raw_data_exttrig:
                raise ValueError('ERROR: Unable to process dIdV. No '
                                 'dIdV data found!')

            if (self._processing_config is not None
                and 'didv' not in self._processing_config):
                raise ValueError(f'ERROR: Input yaml file does '
                                 f'not contain didv processing'
                                 f'configurations!')

            
        if enable_noise:

            if not self._raw_data_notrig:
                raise ValueError('ERROR: Unable to process psd. No '
                                 'randoms or continuous  data found!')

            if (self._processing_config is not None
                and 'noise' not in self._processing_config):
                raise ValueError(f'ERROR: Input yaml file does '
                                 f'not contain noise processing '
                                 f'configurations!')

        if enable_template:
            
            if (self._processing_config is not None
                and 'template' not in self._processing_config):
                raise ValueError(f'ERROR: Input yaml file does '
                                 f'not contain template processing '
                                 f'configurations!')
            

        # when configuration yaml file, certain arguments
        # are disabled
        
        if self._processing_config is not None:

            if channels is not None:
                raise ValueError(f'ERROR: Channels are enabled/disabled '
                                 f'through the yaml file configuration '
                                 f'when provided. Set channels argument '
                                 f'to None!')
            
        # check if file or path
        if (save_file_path is not None
            and not (os.path.isfile(save_file_path)
                     or os.path.isdir(save_file_path))):
            raise ValueError('ERROR: "save_file_path" argument '
                             'should be a file or a path!')


        # check trace length
        if (trace_length_msec is not None
            and trace_length_samples is not None):
            raise ValueError('ERROR: Trace length need to be '
                             'in msec OR samples, nto both')

        if (pretrigger_length_msec is not None
            and pretrigger_length_samples is not None):
            raise ValueError('ERROR: Pretrigger length need to be '
                             'in msec OR samples, nto both')

        #  channels to be processed
        channel_didv = list()
        channel_noise = list()
        channel_template = list()
        
        if channels is not None:
            
            if isinstance(channels, str):
                channels = [channels]


            # check
            for chan in channels:
                
                if (enable_didv and chan not in self._channels_exttrig):
                    raise ValueError(
                        f'ERROR: no {chan} available in dIdV raw data!'
                    )
                
                if (enable_noise and chan not in self._channels_notrig):
                    raise ValueError(
                        f'ERROR: no {chan} available in raw data!'
                    )

            channel_didv = channels.copy()
            channel_noise = channels.copy()
            channel_template = channels.copy()
                
        else:

            # check enable channels 
            #if self._processing_config is not None:
            #   c

            #else:

            if  enable_didv:
                channels = self._channels_exttrig
            elif enable_noise:
                channels = self._channels_notrig
            
                
                
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

            # check IVsweep results available
            for chan in channels:
                
                if chan not in self._calc_true_current:
                    raise ValueError(f'ERROR: Unable to determine if '
                                     f'true current calculation needed '
                                     f'for channel {chan}! Use function: '
                                     f'set_calc_true_current(boolean, '
                                     f'channels="{chan}")')

                if chan not in self._ivsweep_results:
                    raise ValueError(
                        f'ERROR: IVsweep results required! '
                        f'(hdf5 file or results dictionary). Use  '
                        f'function "set_ivsweep_results(...)"')
                   
            # series list 
            series_list = list(self._raw_data_exttrig.keys())

            for chan in channels:

                # calc true current
                calc_true_current = self._calc_true_current[chan]
             
                # ivsweep file
                ivsweep_results = self._ivsweep_results[chan]

                # auto_infinite_lgain
                auto_infinite_lgain = 'auto'
                if chan in self._auto_infinite_lgain:
                    if not self._auto_infinite_lgain[chan]:
                        auto_infinite_lgain = False
                    
                # intitialize output
                output_df = None

                if ncores == 1:
                    output_df = self._process_didv(1,
                                                   series_list,
                                                   chan,
                                                   calc_true_current,
                                                   ivsweep_results,
                                                   auto_infinite_lgain,
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
                                                      repeat(calc_true_current),
                                                      repeat(ivsweep_results),
                                                      repeat(auto_infinite_lgain),
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


            # loop channels
            for chan in channels:
                
                # calc noise psd by series
                if  calc_psd_byseries:
            
                    # intitialize output
                    output_df = None
                    
                    if ncores == 1:
                        output_df = self._process_noise_byseries(
                            1,
                            series_list,
                            chan,
                            nevents,
                            trace_length_msec,
                            pretrigger_length_msec,
                            trace_length_samples,
                            pretrigger_length_samples,
                            psd_amp_freq_ranges)
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
                                repeat(nevents),
                                repeat(trace_length_msec),
                                repeat(pretrigger_length_msec),
                                repeat(trace_length_samples),
                                repeat(pretrigger_length_samples),
                                repeat(psd_amp_freq_ranges)))
                        
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
                                'group_name': self._group_name,
                                'trace_length_samples': trace_length_samples,
                                'pretrigger_length_samples': pretrigger_length_samples}
                    
                    self._filter_data.set_noise_dataframe(chan, output_df,
                                                          metadata=metadata,
                                                          tag=filterdata_tag)
                    
                    # store in dictionary
                    output_dict['noise'][chan] = {'df': output_df}
                    
                # calc noise psd 
                if  calc_psd_global:
                    
                    data = self._process_noise_global(
                        series_list,
                        chan,
                        nevents,
                        trace_length_msec,
                        pretrigger_length_msec,
                        trace_length_samples,
                        pretrigger_length_samples
                    )

                    if chan not in output_dict['noise']:
                        output_dict['noise'][chan] = dict()
                        
                    output_dict['noise'][chan]['psd'] = data['psd']
                    output_dict['noise'][chan]['psd_freqs'] = data['psd_freqs']
                    output_dict['noise'][chan]['psd_fold'] = data['psd_fold']
                    output_dict['noise'][chan]['psd_freqs_fold'] = data['psd_freqs_fold']
                    output_dict['noise'][chan]['sample_rate'] = data['sample_rate']
                    output_dict['noise'][chan]['trace_length_samples'] = (
                        data['trace_length_samples'])
                    output_dict['noise'][chan]['pretrigger_length_samples'] = (
                        data['pretrigger_length_samples'])
                    
                    # add in filter_data
                    metadata = {
                        'processing_id': self._processing_id,
                        'group_name': self._group_name,
                        'nb_samples': data['trace_length_samples'],
                        'nb_pretrigger_samples': data['pretrigger_length_samples'],
                        'sample_rate': data['sample_rate']}
                    
                    # two-sided 
                    self._filter_data.set_psd(chan,
                                              data['psd'],
                                              psd_freqs=data['psd_freqs'],
                                              metadata=metadata,
                                              tag=filterdata_tag)
                    # folded over
                    self._filter_data.set_psd(chan,
                                            data['psd_fold'],
                                            psd_freqs=data['psd_freqs_fold'],
                                            metadata=metadata,
                                            tag=filterdata_tag)
                                    
        # save hdf5
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
                      calc_true_current,
                      ivsweep_results,
                      auto_infinite_lgain,
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
        file_name = ivsweep_results['file']
        iv_type = ivsweep_results['iv_type']
        results =  ivsweep_results['results']

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
                inf_loop_gain_approx=auto_infinite_lgain
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
                                nevents,
                                trace_length_msec,
                                pretrigger_length_msec,
                                trace_length_samples,
                                pretrigger_length_samples,
                                psd_amp_freq_ranges):
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
            
            
        # initialie data dict
        output_dict = {
            'group_name':list(),
            'series_name': list(),
            'psd':list(),
            'psd_freqs':list(),
            'psd_fold':list(),
            'psd_freqs_fold':list(),
            'offset': list(),
            'time_since_group_start': list(),
            'time_since_fridge_run_start':list(),
            'series_time': list()
        }

        config_params = ['tes_bias',
                         'temperature_mc',
                         'temperature_cp',
                         'temperature_still']  

        for param in config_params:
            output_dict[param] = list()
       
        
        # instantiate noise
        noise = Noise(verbose=False)

        # loop series
        for series in series_list:

            if self._verbose:
                print(f'INFO {node_num_str}: Processing noise psd for '
                      f'series {series}!') 
            
            # get files and config
            file_list = self._raw_data_notrig[series]
            if channel not in self._det_config_notrig[series].keys():
                continue
            
            detconfig =  self._det_config_notrig[series][channel]
            seriesconfig = self._series_config_notrig[series]
            
            # for continuous data -> generate randoms
            # for radoms data -> set raw data
            if (trace_length_samples is not None or
                trace_length_msec is not None):
                noise.generate_randoms(file_list, nevents=nevents,
                                       min_separation_msec=100, ncores=1)
            else:
                noise.set_randoms(file_list)


            # calculate noise psd
            noise.calc_psd(
                channel,
                trace_length_msec=trace_length_msec, 
                pretrigger_length_msec=pretrigger_length_msec,
                trace_length_samples=trace_length_samples, 
                pretrigger_length_samples=pretrigger_length_samples,
                nevents=nevents)

            # get psd
            psd, psd_freqs = noise.get_psd(channel)
            psd_fold, psd_freqs_fold = noise.get_psd(channel, fold=True)
        
            # get offset
            offset = noise.get_offset(channel)
            
            # psd ranges
            if psd_amp_freq_ranges is not None:

                # intialize list of parameters
                psd_amp_name_list = list()
                for it, freq_range in enumerate(psd_amp_freq_ranges):

                    # ignore if not a range
                    if len(freq_range) != 2:
                        continue
                    
                    # low/high freqeucy
                    f_low = abs(freq_range[0])
                    f_high = abs(freq_range[1])
                    if f_low > f_high:
                        f_high = f_low + (f_low - f_high)
                        
                    # indices
                    ind_low = np.argmin(np.abs(psd_freqs_fold - f_low))
                    ind_high = np.argmin(np.abs(psd_freqs_fold - f_high))
                    if (ind_low == ind_high):
                        if ind_low < len(psd_freqs_fold)-2:
                            ind_high = ind_low + 1
                        else:
                            continue
                            
                    # take median
                    psd_avg = np.average(psd_fold[ind_low:ind_high])
                    psd_avg = np.sqrt(psd_avg)
                    
                    # parameter name
                    psd_amp_name = ('psd_amp_'
                                    + str(round(f_low))
                                    + '_'
                                    + str(round(f_high)))
            
                    if psd_amp_name in psd_amp_name_list:
                        continue
                    elif psd_amp_name not in output_dict.keys():
                        output_dict[psd_amp_name] = list()

                    # add
                    output_dict[psd_amp_name].append(psd_avg)

            # fill dictionary
            output_dict['series_name'].append(series)
            output_dict['group_name'].append(self._group_name)
            output_dict['psd'].append(psd)
            output_dict['psd_freqs'].append(psd_freqs)
            output_dict['psd_fold'].append(psd_fold)
            output_dict['psd_freqs_fold'].append(psd_freqs_fold)
            output_dict['offset'].append(offset)
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
                              nevents,
                              trace_length_msec,
                              pretrigger_length_msec,
                              trace_length_samples,
                              pretrigger_length_samples):
        
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


        # instantiate noise
        noise = Noise(verbose=True)

        # generate randoms
        data_path = self._base_path + '/' + self._group_name
        if (trace_length_samples is not None or
            trace_length_msec is not None):
            noise.generate_randoms(data_path,
                                   series=series_list,
                                   nevents=nevents,
                                   min_separation_msec=100,
                                   ncores=1)
        else:
            noise.set_randoms(self._base_path,
                              series=series_list)
            
        # calculate noise psd
        noise.calc_psd(
            channel,
            trace_length_msec=trace_length_msec, 
            pretrigger_length_msec=pretrigger_length_msec,
            trace_length_samples=trace_length_samples, 
            pretrigger_length_samples=pretrigger_length_samples,
            nevents=nevents)
        
        # get psd
        psd, psd_freqs = noise.get_psd(channel)
        psd_fold, psd_freqs_fold = noise.get_psd(channel, fold=True)
        sample_rate = noise.get_sample_rate()


        # output dict
        output_dict = {'group_name': self._group_name,
                       'psd':psd,
                       'psd_freqs':psd_freqs,
                       'psd_fold':psd_fold,
                       'psd_freqs_fold':psd_freqs_fold,
                       'sample_rate': sample_rate}
                
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
            config[series] = h5reader.get_detector_config(
                file_name
            )
            info[series] = h5reader.get_file_info(
                file_name)
            for chan in config[series].keys():
                if chan not in channels:       
                    channels.append(chan)
                    
        return info, config, channels
        
    
    def _read_didv_config(self, yaml_file, available_channels):
        """
        Read and check yaml configuration
        file 

        Parameters
        ----------

        yaml_file : str
          yaml configuraton file name (full path)

        available_channels : list
          list of channels available in the file


        Return
        ------
        
        didv_config : dict 
           dictionary with  didv configuration
           
        filter_file : str 
            filter file name (full path) if any
        
        didv_channels : list
            list of all channels to do dIdV
    
        """

        # initialize output
        didv_config = dict()
        filter_file = None
        didv_channels = list()
        
        # open configuration file
        yaml_dict = dict()
        with open(yaml_file) as f:
            yaml_dict = yaml.safe_load(f)
        if not yaml_dict:
            raise ValueError('Unable to read processing configuration!')


        # filter file
        if 'filter_file' not in yaml_dict.keys():
            raise ValueError('ERROR: Filter file required!')
        
        filter_file = yaml_dict['filter_file']
        

        # didv/eventbuilder info
        if 'didv' not in yaml_dict.keys():
            raise ValueError('ERROR: "didv" info required!')

        didv_data = yaml_dict['didv']
        
        
        # Let's loop through keys and find didv channels
        for key, val in didv_data.items():
            
            # Should be a dictionary
            if not isinstance(val, dict):
                continue

            # skip if disable
            if ('run' not in val.keys()
                or ('run' in val.keys()
                    and not val['run'])):
                continue
            
            # we need to split if ',' used 
            if ',' not in key:
                didv_config[key] = val
            else:
                key_split = key.split(',')
                for key_sep in key_split:
                    didv_config[key_sep] = val
                    
            # channel names
            split_char = None
            if ',' in key:
                split_char = ','
            elif '+' in key:
                split_char = '+'
                # elif '-' in key:
                # split_char = '-'
            elif '|' in key:
                split_char = '|'


            if split_char is None:
                didv_channels.append(key)
            else: 
                key_split = key.split(split_char)
                for key_sep in key_split:
                    didv_channels.append(key_sep)
                    
        # Let's check that there are no duplicate channels
        # or channels does not exist 
        channel_list = list()
        for didv_chan in didv_channels:
            
            # check if exist in data
            if didv_chan not in available_channels:
                raise ValueError(
                    'ERROR: didv channel ' + didv_chan
                    + ' does not exist in data')
            
            # check if already didv 
            if didv_chan not in channel_list:
                channel_list.append(didv_chan)
            else:
                print('WARNING: "' + didv_chan 
                      + '" used in multiple didv channels! Will continue '
                      + 'processing regardless, however double check '
                      + 'config file!')
                                        
                    
                    
        # return
        return (didv_config, filter_file, didv_channels)
     

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
