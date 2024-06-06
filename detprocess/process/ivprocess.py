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
import copy
import matplotlib.pyplot as plt
from detprocess.utils import find_linear_segment
from detprocess.core import FilterData

import pytesdaq.io as h5io
import qetpy as qp
from scipy.signal import butter, filtfilt

warnings.filterwarnings('ignore')


__all__ = [
    'IVSweepProcessing'
]



class IVSweepProcessing:
    """
    Class to manage IV/dIdV sweeps processing

    Multiple nodes can be used if data splitted in 
    different series

    """

    def __init__(self, file_path,
                 config_file=None,
                 processing_id=None,
                 bias_tolerance_percent=0.1,
                 verbose=True):
        """
        Intialize data processing 
        
        Parameters
        ---------
    
        file_path : str or list of str
           File(s) or path(s) containing IV and/or dIdV sweep data
                      
        config_file : str 
           Full path and file name to the YAML settings for the
           processing.

        processing_id : str, optional
            an optional processing name. This is used to be build output subdirectory name 
            and is saved as a feature in DetaFrame so it can then be used later during 
            analysis to make a cut on a specific processing when mutliple 
            datasets/processing are added together.

        verbose : bool, optional
            if True, display info


        Return
        ------
        None
        """

        # processing id
        self._processing_id = processing_id

        # display
        self._verbose = verbose

        # Raw file list
        data = self._get_file_list(file_path,
                                   tolerance=bias_tolerance_percent)
        
        data_dict, path_iv, path_didv, name_iv, name_didv =  data

        self._raw_data_dict = data_dict
        self._group_name_iv = name_iv
        self._group_name_didv = name_didv
        self._base_path_iv = path_iv
        self._base_path_didv = path_didv
        
        # bias tolerance
        self._bias_tolerance_percent = bias_tolerance_percent
        self.describe()

        # filter data to store results
        self._filter_data = FilterData()
        
             
    def describe(self):
        """
        Describe data
        """

                
        print(f'\nIV/dIdV sweep available data:')
        
        for chan in self._raw_data_dict.keys():
            print(' ')
            print(f'{chan}:')
                    
            # IV 
            if self._raw_data_dict[chan]['IV'] is not None:
                nb_points = len(
                    list(self._raw_data_dict[chan]['IV'].keys())
                )
                
                print(f' -IV: {nb_points} bias points')

            # dIdV 
            if self._raw_data_dict[chan]['dIdV'] is not None:
                nb_points = len(
                    list(self._raw_data_dict[chan]['dIdV'].keys())
                )
                print(f' -dIdV: {nb_points} bias points')
         
            # common
            if self._raw_data_dict[chan]['IV_dIdV'] is not None:
                nb_points = len(
                    list(self._raw_data_dict[chan]['IV_dIdV'].keys())
                )
                print(f' -Common IV-dIdV: {nb_points} bias points')
            else:
                if (self._raw_data_dict[chan]['IV'] is not None
                    and self._raw_data_dict[chan]['dIdV'] is not None):
                    print(f' -common IV-dIdV: No bias points')


        
                    
    def process(self,
                channels=None,
                enable_iv=True,
                enable_didv=True,
                lgc_output=True,
                lgc_save=False,
                save_path=None,
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
        if not enable_iv and not enable_didv:
            raise ValueError('ERROR: You need to enable IV or dIdV!')
        elif not enable_iv:
            self._group_name_iv = None
            self._base_path_iv = None
        elif not enable_didv:
            self._group_name_didv = None
            self._base_path_didv = None
        
        # check channels
        if channels is None:
            channels = list(self._raw_data_dict.keys())
            if not channels:
                raise ValueError('ERROR: No channels available!')
        elif isinstance(channels, str):
            channels = [channels]
        
        for chan in channels:
            if chan not in self._raw_data_dict.keys():
                raise ValueError(f'ERROR: channel {chan}'
                                 ' not available!')

        # initialize output dictionary
        output_dict = dict()
            
        # Loop channels
        for chan in channels:

            # display
            if self._verbose:
                print(f'INFO: Channel {chan} IV and/or dIdV '
                      'processing')
            
            # data
            channel_series = self._raw_data_dict[chan]
            
            # check if both IV/dIdV:
            if ('IV' not in  channel_series.keys()
                or channel_series['IV'] is None):
                enable_iv = False
            if ('dIdV' not in  channel_series.keys()
                or channel_series['dIdV'] is None):
                enable_didv = False
                
            # processing type
            processing_type = str()
            if enable_iv and enable_didv:
                if 'IV_dIdV' in  channel_series.keys():
                    processing_type = 'IV_dIdV'
                else:
                    raise ValueError(
                        f'ERROR: Unable to process both IV and dIdV for ' 
                        'channel {chan}. No common bias points. If need, '
                        'tolerance can be modified with "bias_tolerance_percent" '
                        'during IVSweepProcessing instantiation!')   
            elif (enable_iv and 'IV' in  channel_series.keys()):
                processing_type = 'IV'
            elif (enable_didv and 'dIdV' in  channel_series.keys()):
                processing_type = 'dIdV'
            else:
                raise ValueError('ERROR: No IV or dIdV found. '
                                 'Check data availability and setup.')

            # get list of bias points
            bias_points = list(channel_series[processing_type].keys())
                       
            # check number cores allowed
            if ncores>len(bias_points):
                ncores = len(bias_points)
                if self._verbose:
                    print('INFO: Changing number cores to '
                          + str(ncores) + ' (maximum allowed)')
            
            # case only 1 node used for processing
            output_channel_df = None
            if ncores == 1:
                output_channel_df = self._process(
                    1, chan, processing_type, bias_points
                )
            else:
                # split data
                bias_points_split = self._split_series(bias_points,
                                                       ncores)
                # launch
                node_nums = list(range(ncores+1))[1:]
                pool = Pool(processes=ncores)
                df_list = pool.starmap(
                    self._process,
                    zip(node_nums,
                        repeat(chan),
                        repeat(processing_type),
                        bias_points_split)
                )
                pool.close()
                pool.join()
                    
                # concatenate
                output_channel_df =  pd.concat(df_list,
                                               ignore_index=True)
         
            # sort based on bias
            output_channel_df = output_channel_df.sort_values(
                'tes_bias', ascending=False, key=abs,
                ignore_index=True
            )

            # add state (normal, SC, in transition)
            output_channel_df['state'] = np.nan

            # check if SC/Normal points based on linearity
            tes_bias = output_channel_df['tes_bias'].values
            offset = None
            if 'offset_noise' in output_channel_df.columns:
                offset = output_channel_df['offset_noise'].values
            else:
                offset = output_channel_df['offset_didv'].values
                
            # normal
            normal_indices = np.array(find_linear_segment(tes_bias, offset))
            nb_normal_points = len(normal_indices)
            if nb_normal_points>0:
                output_channel_df['state'].iloc[normal_indices] = 'normal'
                
            # normal
            tes_bias = np.flip(tes_bias).copy()
            offset = np.flip(offset).copy()
            sc_indices = np.array(find_linear_segment(tes_bias, offset))
            nb_sc_points = len(sc_indices)
            if nb_sc_points>0:
                sc_indices = len(tes_bias) - sc_indices - 1
                output_channel_df['state'].iloc[sc_indices] = 'sc'
                            
            # save  output dataframe in dictionary
            output_dict[chan] = output_channel_df

            if self._verbose:
                print(f'INFO: IV/dIdV processing done for channel {chan}!')

                if nb_sc_points > 0:
                    print(f'INFO: Found {nb_sc_points} SC points based on linearity')
                else:
                    print('INFO: Unable to estimate number of SC points '
                          'based on linearity. Check data!')
                if nb_normal_points > 0:
                    print(f'INFO: Found {nb_normal_points} normal points based on linearity')
                else:
                    print('INFO: Unable to estimate number of normal points '
                          'based on linearity. Check data!')
                    
        # save all dataframes in filter data
        self._filter_data.set_ivsweep_data_from_dict(output_dict)

            
        if lgc_save:

            if save_path is None:
                save_path = self._base_path_didv
                if (self._base_path_iv and self._base_path_iv is not None):
                    save_path = self._base_path_iv
                save_path  = save_path + '/filterdata'
                if '/raw/filterdata' in save_path:
                    save_path = save_path.replace('/raw/filterdata',
                                                  '/filterdata')

            group_name =  self._group_name_didv
            if (self._group_name_iv and self._group_name_iv is not None):
                 group_name = self._group_name_iv

            if group_name not in save_path:
                save_path = save_path + '/' + group_name 

            if not os.path.isdir(save_path):
                try:
                    os.makedirs(save_path)
                except OSError:
                    raise ValueError(f'\nERROR: Unable to create directory '
                                     f'"{save_path}"!')
                       
            now = datetime.now()
            series_day = now.strftime('%Y') +  now.strftime('%m') + now.strftime('%d') 
            series_time = now.strftime('%H') + now.strftime('%M')
            series_name = ('D' + series_day + '_T'
                           + series_time + now.strftime('%S'))
            if self._processing_id is not None:
                file_name = (save_path + '/'
                             + self._processing_id
                             + '_' + series_name + '.hdf5')
            else:
                file_name = (save_path + '/ivsweep_processing_'
                             + series_name + '.hdf5')

            self._filter_data.save_hdf5(file_name)
            print(f'INFO: Saving dataframe in {file_name}') 

        if self._verbose:
            print('INFO: IV/dIdV processing done!') 
            
        if lgc_output:
            return output_dict
        

        
    def plot_ivsweep_offset(self, channel, tag='default'):
        """
        Plot offset vs tes_bias with errors from IV and if available
        dIdV offset
        """
        self._filter_data.plot_ivsweep_offset(channel=channel,
                                              tag=tag)
        
        
    def _process(self, node_num,
                 channel,
                 processing_type,
                 bias_list):
                 
        """
        Process data
        
        Parameters
        ---------

        node_num :  int
          node id number, used for display
       
        processing_type : str
          processing type ('IV' or 'dIdV')

 
        series_list : str
          list of series name to be processed
        
          
        """

        # node string (for display)
        node_num_str = str()
        if node_num>-1:
            node_num_str = 'Node #' + str(node_num)
            
        # data list
        data_dict = self._raw_data_dict[channel][processing_type]
        
        # intialize data dictionary
        output_data = {'channel':list(),
                       'tes_bias_uA' : list(),
                       'tes_bias' : list(),
                       'rshunt' : list(),
                       'temperature_mc': list(),
                       'temperature_cp': list(),
                       'temperature_still': list()}
        
        # data types
        data_types = list()
        if (processing_type=='IV'
            or processing_type=='IV_dIdV'):
            data_types.append('noise')
        if (processing_type=='dIdV'
            or processing_type=='IV_dIdV'):
            data_types.append('didv')

            
        for ptype in data_types:

            ptype_output_data = {
                'series_name_' + ptype: list(),
                'group_name_' + ptype: list(),
                'fs_' + ptype: list(),
                'output_variable_gain_' + ptype: list(),
                'output_variable_offset_' + ptype: list(),
                'close_loop_norm_' + ptype: list(),
                'rshunt_' + ptype: list(),
                'tes_bias_' + ptype: list(),
                'offset_' + ptype: list(),
                'offset_err_' + ptype: list(),
                'cut_eff_' + ptype: list(),
                'cut_' + ptype: list(),
                'cut_pass_' + ptype: list(),
                'avgtrace_' + ptype: list()
            }
            
            if ptype == 'noise':
                ptype_output_data['psd'] = list()
                ptype_output_data['psd_freq'] = list()
            else:
                ptype_output_data['sgamp'] = list()
                ptype_output_data['sgfreq'] = list()
                ptype_output_data['didvmean'] = list()
                ptype_output_data['didvstd'] = list()
                ptype_output_data['dutycycle'] = list()
                ptype_output_data['rtes_estimate'] = list()

            output_data.update(ptype_output_data)

            
        # loop series
        for bias in bias_list:

            # verbose
            if self._verbose:
                print(f'INFO {node_num_str}: processing bias point {bias} uA')

            # store general parameters
            output_data['channel'].append(channel)
            output_data['tes_bias_uA'].append(bias)

            # data bias dict
            file_dict = data_dict[bias]
            
            # Loop IV / dIdV
            for itype in range(len(data_types)):
                
                ptype = data_types[itype]
                
                if processing_type == 'IV_dIdV':
                    if ptype == 'noise':
                        file_list = file_dict['IV']
                    else:
                        file_list = file_dict['dIdV']
                else:
                    file_list = file_dict
            
                # check files
                if not file_list:
                    raise ValueError(f'ERROR: IV/dIdV processing: '
                                     'No files found for bias '
                                     'point {bias}uA ,'
                                     'channel {channel} ')
                
                # load data
                traces = None
                detector_settings = None
                fs = None
                
                try:
                    h5 = h5io.H5Reader()
                    traces, info = h5.read_many_events(
                        filepath=file_list,
                        output_format=2,
                        include_metadata=True,
                        detector_chans=channel,
                        adctoamp=True)

                    traces = traces[:,0,:]
                    fs  = info[0]['sample_rate']
                    detector_settings = h5.get_detector_config(file_name=file_list[0])
                    del h5
                    
                except:
                    raise OSError('ERROR:Unable to get traces or detector '
                                  'settings from hdf5 data!')

               
                # detector parameters
                tes_bias = float(detector_settings[channel]['tes_bias'])
                output_gain = float(detector_settings[channel]['output_gain'])
                close_loop_norm = float(detector_settings[channel]['close_loop_norm'])
                output_offset = float(detector_settings[channel]['output_offset'])
                sgamp = float(detector_settings[channel]['signal_gen_current'])
                sgfreq = float(detector_settings[channel]['signal_gen_frequency'])
                rshunt = float(detector_settings[channel]['shunt_resistance'])
                group_name = str(info[0]['group_name'])
                series_name = h5io.extract_series_name(int(info[0]['series_num']))
                dutycycle = 0.5
                if 'dutycycle' in detector_settings[channel]:
                    dutycycle = float(detector_settings[channel]['dutycycle'])


                # store readout data
                    
                # temperature
                if itype == 0:
                    temperature_list = ['cp','mc','still']
                    for temp in temperature_list:
                        temp_par = 'temperature_' + temp
                        temp_val = np.nan
                        if temp_par in  detector_settings[channel]:
                            temp_val = float(detector_settings[channel][temp_par])
                        output_data[temp_par].append(temp_val)
                      
                                  
                # store readout data
                if itype == 0:
                    output_data['tes_bias'].append(tes_bias)
                    output_data['rshunt'].append(rshunt)
                    
                output_data['series_name_' + ptype].append(series_name)
                output_data['group_name_' + ptype].append(group_name)
                output_data['fs_' + ptype].append(fs)
                output_data['output_variable_gain_' + ptype].append(output_gain)
                output_data['output_variable_offset_' + ptype].append(output_offset)
                output_data['close_loop_norm_' + ptype].append(close_loop_norm)
                output_data['rshunt_' + ptype].append(rshunt)
                output_data['tes_bias_' + ptype].append(tes_bias)
           
                if ptype == 'noise':

                    # get rid of traces that are all zero
                    zerocut = np.all(traces!=0, axis=1)
                    traces = traces[zerocut]
                    
                    # apply autocut
                    cut = qp.autocuts_noise(traces, fs=fs)
                    cut_pass = True
                    cut_eff = np.sum(cut)/len(cut)
                    traces = traces[cut]
                
                    # PSD calculation
                    psd_freq, psd = qp.calc_psd(traces, fs=fs,
                                                folded_over=False)
                    
                    # Offset calculation
                    offset, offset_err = qp.utils.calc_offset(traces, fs=fs)
                    
                    # Pulse average
                    avgtrace = np.mean(traces, axis=0)

                    # store psd
                    output_data['psd'].append(psd)
                    output_data['psd_freq'].append(psd_freq)

                else:

                    # get rid of traces that are all zero
                    zerocut = np.all(traces!=0, axis=1)
                    traces = traces[zerocut]
                    
                    # pile-up cuts
                    cut = qp.autocuts_didv(traces, fs=fs)
                    cut_pass = True
                    cut_eff = np.sum(cut)/len(cut)
                    traces = traces[cut]
                    
                    # Offset calculation
                    offset, offset_err = qp.utils.calc_offset(
                        traces, fs=fs, sgfreq=sgfreq, is_didv=True)
                
                    # Average pulse
                    avgtrace = np.mean(traces, axis=0)

                    # get estimated resistance (using 75% single cycle)
                    avgtrace_lp = qp.utils.lowpassfilter(avgtrace,
                                                         cut_off_freq=1.5e3,
                                                         fs=fs, order=1)

                    nb_bins = len(avgtrace_lp)
                    nb_cycles = (nb_bins/fs)*sgfreq
                    idx_calc = range(round(nb_bins*0.1),
                                     round(nb_bins*0.9))
                    if nb_cycles>1.5:
                        start_bin = round(fs/sgfreq*0.25)
                        end_bin = 4*start_bin
                        idx_calc = range(start_bin, end_bin)
                    deltaI =  avgtrace_lp[idx_calc].max() - avgtrace_lp[idx_calc].min()
                    deltaV = sgamp*rshunt
                    rtes_estimate = deltaV/deltaI*1e3
                                     
                    # dIdV fit
                    didvobj = qp.DIDV(
                        traces,
                        fs,
                        sgfreq,
                        sgamp,
                        rshunt,
                        autoresample=False,
                        dutycycle=dutycycle,
                    )
                
                    didvobj.processtraces()

                    # store didv specific data
                    output_data['sgamp'].append(sgamp)
                    output_data['sgfreq'].append(sgfreq)
                    output_data['didvmean'].append(didvobj._didvmean)
                    output_data['didvstd'].append(didvobj._didvstd)
                    output_data['dutycycle'].append(dutycycle)
                    output_data['rtes_estimate'].append(rtes_estimate)

                # store common data
                output_data['offset_' + ptype].append(offset)
                output_data['offset_err_' + ptype].append(offset_err)
                output_data['cut_eff_' + ptype].append(cut_eff)
                output_data['cut_' + ptype].append(cut)
                output_data['cut_pass_' + ptype].append(cut_pass)
                output_data['avgtrace_' + ptype].append(avgtrace)
                          
        # convert to dataframe
        df = pd.DataFrame.from_dict(output_data)
        return df
            
                
           
        
    def _get_file_list(self, file_path, tolerance=0.1):
        
        """
        Get file list from path. Return as a dictionary
        with key=series and value=list of files

        Parameters
        ----------

        file_path : str or list of str
           File(s) or path(s) containing IV and/or dIdV sweep data
                      

        Return
        -------
        
        data_dict : dict 
          list of files for splitted inot series
     
        group_name : str
           group name of raw data

        """

        if self._verbose:
            print('INFO: Checking sweep data. Be patient!')

        # get list of files
        if isinstance(file_path, str):
            file_path = [file_path]
            
            
        # initialize
        file_list = list()
        base_path_iv = list()
        base_path_didv = list()
        group_name_iv = list()
        group_name_didv = list()

        # loop files 
        for a_path in file_path:

            # case path is a directory
            if os.path.isdir(a_path):
                file_list.extend(glob(a_path + '/*_F0001.hdf5'))
                                  
            # case file
            elif os.path.isfile(a_path):
                file_list.append(a_path)
            else:
                raise ValueError('File or directory "' + a_path
                                 + '" does not exist!')
            
        if not file_list:
            raise ValueError('ERROR: No raw data found. Check arguments!')
        file_list.sort()
    
        # initialize raw reader
        h5reader = h5io.H5Reader()


        # 1. split IV/dIdV series
        series_dict = {'IV': None, 'dIdV': None}
        for a_file in file_list:

            metadata = h5reader.get_file_info(a_file)
            
            # data purpose
            data_purpose = metadata['data_purpose']
            
            # series name
            series_name = h5io.extract_series_name(metadata['series_num'])
            
            # file list
            raw_path = str(Path(a_file).parents[0])
            series_file_list = glob(f'{raw_path}/*{series_name}_F*.hdf5')
            
            if data_purpose == 'IV':
                if series_dict['IV'] is None:
                    series_dict['IV'] = dict()
                series_dict['IV'][series_name] = series_file_list
                base_path_iv.append(str(Path(a_file).parents[1]))
                group_name_iv.append(str(Path(Path(a_file).parent).name))
            elif data_purpose == 'dIdV':
                if series_dict['dIdV'] is None:
                    series_dict['dIdV'] = dict()
                series_dict['dIdV'][series_name] = series_file_list
                base_path_didv.append(str(Path(a_file).parents[1]))
                group_name_didv.append(str(Path(Path(a_file).parent).name))
            else:
                raise ValueError(f'ERROR: Unknow data purpose "{data_purpose}"')


        # check base_path / group
        base_path_iv = list(set(base_path_iv))
        base_path_didv = list(set(base_path_didv))
        group_name_iv = list(set(group_name_iv))
        group_name_didv = list(set(group_name_didv))
        
        if  (len(base_path_iv)>1 or len(group_name_iv)>1):
            raise ValueError('ERROR: IV data should be in a single directory!')
        if  (len(base_path_didv)>1 or len(group_name_didv)>1):
            raise ValueError('ERROR: dIdV data should be in a single directory!')

        if group_name_iv:
            base_path_iv = base_path_iv[0]
            group_name_iv = group_name_iv[0]
        if group_name_didv:
            base_path_didv =  base_path_didv[0]
            group_name_didv = group_name_didv[0]
            
        # 2. find sweep channels
        channels = list()
        channel_dict = {'IV':list(), 'dIdV':list()}
        data_types = ['IV', 'dIdV']
        for data_type in data_types:
            
            # check if data available
            if series_dict[data_type] is None:
                continue
            
            # first/last config
            series_list = list(series_dict[data_type].keys())
         
            file_first = series_dict[data_type][series_list[0]][0]
            file_last = series_dict[data_type][series_list[-1]][0]
         
            config_first = h5reader.get_detector_config(file_first)
            config_last = h5reader.get_detector_config(file_last)
                                                       
            # loop channels
            for chan in config_first.keys():
                bias_first = round(float(config_first[chan]['tes_bias'])*1e6, 3)
                bias_last = round(float(config_last[chan]['tes_bias'])*1e6, 3)
                
                if (abs(bias_first-bias_last)/abs(max(bias_first, bias_last)) > 0.001):
                    channels.append(chan)
                    if data_type=='IV':
                        channel_dict['IV'].append(chan)
                    else:
                        channel_dict['dIdV'].append(chan)
                        
                        
        # 3. find series for each channel
 
        # initialize
        data_dict = dict()
        for chan in channels:
            data_dict[chan] = {'IV': None, 'dIdV': None}
            
        
        # For IV data, all sweep channels are from same series
        for chan in channel_dict['IV']:
            data_dict[chan]['IV'] = copy.deepcopy(series_dict['IV'])

        # For dIdV, if more than 1 channel
        # we need to check if signal generator on/off
        # FIXME: add file attributes
        if len(channel_dict['dIdV']) == 1:
            data_dict[chan]['dIdV'] = copy.deepcopy(series_dict['dIdV'])
            
        elif len(channel_dict['dIdV'])>1:
            
            # loop series
            for series_name in list(series_dict['dIdV'].keys()):
                
                file_name = glob(f'{raw_path}/*_{series_name}_F0001.hdf5')[0]
                
                # get detector config
                config = h5reader.get_detector_config(file_name)

                # check signal generator
                for chan in channel_dict['dIdV']:
                    
                    # FIXME: Temporary fix for LBL (assume using "starcryo")
                    is_didv = ((config[chan]['signal_gen_onoff']=='on' 
                                and config[chan]['signal_gen_source']=='tes') or
                               (config[chan]['signal_gen_onoff']=='on'
                                and 'starcryo' in config[chan]['controller_chans']))
                    
                    if is_didv:
                        
                        if data_dict[chan]['dIdV'] is None:
                            data_dict[chan]['dIdV'] = dict()
                            
                        data_dict[chan]['dIdV'][series_name] = (
                            copy.deepcopy(series_dict['dIdV'][series_name])
                        )


        # 4) find tes bias  for each series/channels
        for chan in  data_dict.keys():
            for ptype in data_dict[chan].keys():
                
                # check if type available
                if data_dict[chan][ptype] is None:
                    continue
                
                # loop series and make new dictionary
                bias_dict = dict()
                ptype_dict = data_dict[chan][ptype]
                for series, series_list in ptype_dict.items():
                    config = h5reader.get_detector_config(series_list[0])
                    tes_bias = round(float(config[chan]['tes_bias'])*1e6, 3)
                    bias_dict[tes_bias] = series_list

                # replace
                data_dict[chan][ptype] = bias_dict

        # check common bias
        for chan, chan_dict in data_dict.items():

            # initialize
            data_dict[chan]['IV_dIdV'] = None
            if (chan_dict['IV'] is None
                or chan_dict['dIdV'] is None):
                continue

            iv_bias_points = list(chan_dict['IV'].keys())
            didv_bias_points = list(chan_dict['dIdV'].keys())

            # Initialize an array to hold the common elements
            data_dict[chan]['IV_dIdV'] = dict()

            # tolerance
            tolerance = tolerance / 100

            # Loop iv 
            for iv_bias in iv_bias_points:
                for didv_bias in didv_bias_points:
                    if (np.abs(iv_bias-didv_bias)/np.abs(
                            max(iv_bias, didv_bias)) <= tolerance):
                        data_dict[chan]['IV_dIdV'][iv_bias] = {
                            'IV': chan_dict['IV'][iv_bias],
                            'dIdV': chan_dict['dIdV'][didv_bias]
                        }
                        
        return (data_dict, base_path_iv, base_path_didv,
                group_name_iv, group_name_didv)
    

    

    def _create_output_directory(self, base_path, facility):
        
        """
        Create output directory 

        Parameters
        ----------
        
        base_path :  str
           full path to base directory 
        
        facility : int
           id of facility 
    
        Return
        ------
          output_dir : str
            full path to created directory

        """

        now = datetime.now()
        series_day = now.strftime('%Y') +  now.strftime('%m') + now.strftime('%d') 
        series_time = now.strftime('%H') + now.strftime('%M')
        series_name = ('I' + str(facility) +'_D' + series_day + '_T'
                       + series_time + now.strftime('%S'))

        series_num = h5io.extract_series_num(series_name)
        
        # prefix
        prefix = 'feature'
        if self._processing_id is not None:
            prefix = self._processing_id + '_feature'
        if restricted:
            prefix += '_restricted'
        output_dir = base_path + '/' + prefix + '_' + series_name
        
        
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
                os.chmod(output_dir, stat.S_IRWXG | stat.S_IRWXU | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'+ output_dir  + '"!\n')
                
        return output_dir, series_num
        

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
