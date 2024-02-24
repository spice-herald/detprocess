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

    def process(self,
                channels=None,
                enable_psd=False,
                enable_didv=False,
                nevents=None,
                processing_id=None,
                lgc_output=False,
                lgc_save=False,
                save_path=None,
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

        save_path : str, optional
           base directory where output group will be saved
           default: same base path as input data
    
        ncores: int, optional
           number of cores that will be used for processing
           default: 1
        """
        
        # check enable IV, dIdV
        if not enable_didv and not enable_psd:
            raise ValueError('ERROR: You need to enable dIdV or PSD calculation!')

        if enable_didv and enable_psd:
            raise ValueError('ERROR: Enable dIdV or PSD, not both!')
        
        # check channels
        if channels is not None:
            
            if isinstance(channels, str):
                channels = [channels]
                
            for chan in channels:
                if (enable_didv and chan not in self._channels_exttrig):
                    raise ValueError(
                        f'ERROR: no {chan} available in dIdV raw data!'
                    )
                
                if (enable_psd and chan not in self._channels_notrig):
                    raise ValueError(
                        f'ERROR: no {chan} available in raw data!'
                    )
                
        else:        
            if  enable_didv:
                channels = self._channels_exttrig
            elif enable_psd:
                channels = self._channels_notrig
                
        if enable_psd:
            nseries = len(self._det_config_notrig.keys())
            if ncores > nseries:
                ncores = nseries
                
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
                output_dict['didv'][chan] = output_df
                
        # save
        if lgc_save:

            # save path
            if save_path is None:
                save_path = self._base_path
                save_path += '/filterdata'
                if '/raw/filterdata' in save_path:
                    save_path = save_path.replace('/raw/filterdata',
                                                  '/filterdata')
            # add group name
            if self._group_name not in save_path:
                save_path = save_path + '/' + self._group_name

            # create file name
            file_name = self._create_file_name(
                save_path, self._facility, self._processing_id
            )

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
        data_types = ['rand', 'cont', 'exttrig',
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
            elif 'threshtrig' in file_name:
                file_type = 'threshtrig' 

            if file_type is None:
                file_info = h5reader.get_file_info(afile)
                if 'data_type' in file_info:
                    data_type = int(file_info['data_type'])
                    if (data_type == 1 or data_type == 3):
                        file_type = 'notrig'
                    elif data_type == 2:
                        file_type = 'exttrig'
                    elif data_type == 4:
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
     

    def _create_file_name(self, save_path, facility, processing_id):
        
        """
        Create output directory 

        Parameters
        ----------
        
        save_path :  str
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
        if not os.path.isdir(save_path):
            try:
                os.makedirs(save_path)
                os.chmod(output_dir,
                         stat.S_IRWXG | stat.S_IRWXU
                         | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'
                                 + save_path  + '"!\n')

        # build file name
        file_name = save_path + '/' + prefix + '_' + series_name + '.hdf5'
                
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


