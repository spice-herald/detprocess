import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io
import qetpy as qp
from glob import glob
import vaex as vx
from pathlib import Path
from detprocess.core.filterdata import FilterData

__all__ = [
    'IVSweepAnalysis'
]


class IVSweepAnalysis(FilterData):
    """
    Class to manage iv/sweep calculations using 
    QETpy
    """

    def __init__(self, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information

        """

        # IV, DIDV, Noise, Template objects    
        self._iv_objects = dict()
        self._didv_objects = dict()
        self._noise_objects = dict()
        self._template_objects = dict()


        # rshunt/rshunt_err if not in raw data
        # rp/rp error if no SC points
        self._readout_params = dict()
        
        
        # instantiate base class
        super().__init__(verbose=verbose)
        
        
    def clear(self, channels=None):
        """
        Clear all data
        """
        # objects
        self._iv_objects = dict()
        self._didv_objects = dict()
        self._noise_objects = dict()
        self._template_objects = dict()
        
        # filter data
        self.clear_data()


    def set_data_from_file(self, file_name):
        """
        Load IV processed data from file
        """

        self.load_hdf5(file_name)
        self.describe()


    def set_data_from_dict(self, data_dict, tag='default'):
        """
        Set data using dictionary  with 
        keys=channels, values=dataframe
        """

        for chan, df in data_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    'ERROR: Unrecognized format. Dictionary '
                    'item should be a pandas dataframe!'
                )
            self.set_data_from_dataframe(chan, df, tag=tag)
            

        self.describe()

    def set_data_from_dataframe(self, channel, df,
                                tag='default'):
        """
        Set data from dataframe for a specific channel
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                'ERROR: Unrecognized format. Dictionary '
                'item should be a pandas dataframe!'
            )

        # sort
        df = df.sort_values('tes_bias',
                            ascending=False,
                            key=abs,
                            ignore_index=True)
        
        # store
        self.set_ivsweep_data(channel,
                              df,
                              metadata=None,
                              tag=tag)
        
        self.describe()

        
    
    def set_rshunt(self, channels,
                   rshunt=None, rshunt_err=None):
        """
        Set rshunt and/or rshunt error for one or more channels
        if not stored in raw data file.
 
        If multiple channels, then all arguments should be a list. 
        """


        # convert to list 
        if isinstance(channels, str):
            channels = [channels]

        params = {'rshunt': rshunt,
                  'rshunt_err': rshunt_err}

        for param, param_val in params.items():
            if param_val is not None:
                if  (isinstance(param_val, int)
                     or isinstance(param_val, float)) :
                    param_val = [param_val]
                if len(param_val) == 1:
                    param_val = param_val*len(channels)
                elif len(param_val)!=len(channels):
                    raise ValueError(
                        f'ERROR: {param} list should be have length = '
                        ' # channels')
            else:
                param_val = [None]*len(channels)

            # replace
            params[param] =  param_val
      
            
        for ichan, chan in enumarate(channels):
            if chan not in self._readout_params.keys():
                self._readout_params[chan] = dict()
            self._readout_params[chan]['rshunt'] = params['rshunt'][ichan]
            self._readout_params[chan]['rshunt_err'] = params['rshunt_err'][ichan]


    def set_rp(self, channels,
               rp=None, rp_err=None):
        """
        Set rp and/or rp error for one or more channels if no SC points.  
        If multiple channels, then all arguments should be a list. 
        """


        # convert to list 
        if isinstance(channels, str):
            channels = [channels]


        params = {'rp': rp,
                  'rp_err': rp_err}

        for param, param_val in params.items():
            if param_val is not None:
                if  (isinstance(param_val, int)
                     or isinstance(param_val, float)):
                    param_val = [param_val]
                if len(param_val) == 1:
                    param_val = param_val*len(channels)
                elif len(param_val)!=len(channels):
                    raise ValueError(
                        f'ERROR: "{param}" list should be have length = '
                        ' # channels')
            else:
                param_val = [None]*len(channels)

            # replace
            params[param] = param_val
            
        for ichan,chan in enumarate(channels):
            if chan not in self._readout_params.keys():
                self._readout_params[chan] = dict()
            self._readout_params[chan]['rp'] =  params['rp'][ichan]
            self._readout_params[chan]['rp_err'] =  params['rp_err'][ichan]



            
    def analyze_sweep(self, channels=None, nnorms=None, nscs=None,
                      lgc_invert_offset='auto',
                      lgc_plot=False, lgc_save_plot=False,
                      plot_save_path=None,
                      tag='default',
                      **kwargs):
        """
        Analyze IV sweep (multi-channels) to  calculate I0,R0,P0 
        in function of bias.
        """

        # --------------------------------
        # Check arguments
        # ---------------------------------
        
        # check data
        available_channels = list()
        ivdata_label = 'ivsweep_data_' + tag
        for chan, chan_dict in self._filter_data.items():
            if ivdata_label in chan_dict.keys():
                available_channels.append(chan)

        if not available_channels:
            raise ValueError('ERROR: No channel available. '
                             'Set data first!')
        if channels is None:
            channels = available_channels
        else:
            if isinstance(channels, str):
                channels = [channels]
            for chan in channels:
                if chan not in available_channels:
                    raise ValueError(f'ERROR: No data for channel {chan}'
                                     ' available. Set data first!')
        # number of channels
        nb_channels = len(channels)

        # check number of normal / SC
        if nnorms is not None:
            if isinstance(nnorms, int):
                nnorms = [nnorms]
            elif len(nnorms) != nb_channels:
                raise ValueError(
                    'ERROR: "nnorms" should have same length'
                    'as "channels"')
        else:
            nnorms = [None]*nb_channels

        if nscs is not None:
            if isinstance(nscs, int):
                nscs = [nscs]
            elif len(nscs) != nb_channels:
                raise ValueError(
                    'ERROR: "nscs" should have same length'
                    'as "channels"')
        else:
            nscs = [None]*nb_channels

        # --------------------------------
        # Check sweep data for all channels
        # first
        # ---------------------------------
        
        # loop channel and check if sc / normal points available
        # check Rshunt and Rp (needed if no SC points)
        for ichan, chan in enumerate(channels):
            
            df = self.get_ivsweep_data(chan, tag=tag)
            df = df.sort_values('tes_bias',
                                ascending=False,
                                key=abs,
                                ignore_index=True)

            
            # check processing has been done
            if ('offset_noise' not in df.columns
                and 'offset_didv' not in df.columns):
                raise ValueError(f'ERROR: Unable to find an offset in'
                                 ' the channel {chan} dataframe! Check data!')
            
            # Rp / Rshunt
            rp = None
            rp_err = None
            rshunt = None
            rshunt_err = None

            # first check if parameters have been set manually
            for param_chan, param_dict in self._readout_params.items():
                if chan == param_chan:
                    if 'rp' in  param_dict.keys():
                        rp = param_dict['rp']
                    if 'rp_err' in  param_dict.keys():
                        rp_err = param_dict['rp_err']
                    if 'rshunt' in  param_dict.keys():
                        rshunt = param_dict['rshunt']
                    if 'rshunt_err' in  param_dict.keys():
                        rshunt_err = param_dict['rshunt_err']


            # If None, use dataframe
            if rp is None:
                if ('rp' in df.columns
                    and df['rp'].values[0] != np.nan):
                    rp = df['rp'].values[0]
            if rp_err is None:
                if ('rp_err' in df.columns
                    and df['rp_err'].values[0] != np.nan):
                    rp_err = df['rp_err'].values[0]
                elif rp is not None:
                    rp_err = rp*0.02
            if rshunt is None:
                if ('rshunt' in df.columns
                    and df['rshunt'].values[0] != np.nan):
                    rshunt = df['rshunt'].values[0]
            if rshunt_err is None:
                if ('rshunt_err' in df.columns
                    and df['rshunt_err'].values[0] != np.nan):
                    rshunt_err = df['rshunt_err'].values[0]
                elif rshunt is not None:
                    rshunt_err = rshunt*0.02
                    
            # store parameters
            if chan not in self._readout_params.keys():
                self._readout_params[chan] = dict()
            self._readout_params[chan]['rp'] = rp
            self._readout_params[chan]['rp_err'] = rp_err
            self._readout_params[chan]['rshunt'] = rshunt
            self._readout_params[chan]['rshunt_err'] = rshunt_err
            
                           
            # normal points
            nnorm = nnorms[ichan]
            if nnorm is None:
                normal_idx = np.where(df['state']=='normal')[0]
                if len(normal_idx) == 0:
                    raise ValueError(
                        'ERROR: Unknow number of normal points for '
                        'channel {chan}. Add "nnorms" argument!')
                else:
                    nnorms[ichan] = len(normal_idx)
                    
            # SC points
            nsc = nscs[ichan]
            if nsc is None:
                nsc_idx = np.where(df['state']=='sc')[0]
                if len(nsc_idx) == 0:
                    raise ValueError(
                        f'ERROR: Unknow number of SC points for channel {chan}. '
                        'To disable SC, set Rp and Rp error using "set_rp()" '
                        'function and set "nscs=0" argument'
                    )
                else:
                    nscs[ichan] = len(nsc_idx)
            elif nsc == 0:
                if rp is None or rp_err is None:
                    raise ValueError(
                        f'ERROR: Rp is required for channel {chan} when nscs=0 '
                        'Use set_rp() function first!')
                
            # check rshunt
            if rshunt is None or rshunt_err is None:
                raise ValueError(
                    f'ERROR: Rshunt is required for channel {chan}! '
                    'Use set_rshunt() function first!'
                )
            

        # --------------------------------
        # Loop channel and do analysis
        # ---------------------------------
        # Loop channels
        for ichan, chan in enumerate(channels):
            
            # get dataframe
            df = self.get_ivsweep_data(chan, tag=tag)
            df = df.sort_values('tes_bias',
                                ascending=False,
                                key=abs,
                                ignore_index=True)

            
            # sc/normal range
            nnorm = nnorms[ichan]
            range_normal = range(nnorm)
            
            range_sc = []
            nsc = nscs[ichan]
            fitsc = False
            if nsc > 0:
                fitsc = True
                range_sc = range(len(df)-nsc,
                                 len(df))
            
            # parameters dictionary
            param_dict = self._readout_params[chan]

            # check noise/dIdV sweep
            data_types = list()
            labels = list()
            if 'offset_noise' in df.columns:
                data_types.append('noise')
                labels.append(f'{chan} noise')
            if 'offset_didv' in df.columns:
                data_types.append('didv')
                labels.append(f'{chan} dIdV')
                
            # get values from dataframe
            tes_bias = df['tes_bias'].values
            offset = df['offset_' + data_types[0]].values
            offset_err = df['offset_err_' + data_types[0]].values
            if len(data_types)==2:
                tes_bias = np.vstack([tes_bias, tes_bias])
                tes_bias[1,:] = df['tes_bias_' + data_types[1]].values
                offset = np.vstack([offset, offset])
                offset[1,:] =  df['offset_' + data_types[1]].values
                offset_err = np.vstack([offset_err, offset_err])
                offset_err[1,:] =  df['offset_err_' + data_types[1]].values


            # invert offset
            offset = offset.copy()
            if (lgc_invert_offset != 'auto'
                and lgc_invert_offset):
                offset = -1*offset
                
            # rp guess
            rp = param_dict['rp']
            rp_err= param_dict['rp_err']
            if rp is None:
                rp = 0
                rp_err = 0
                
            # instantiate
            ivobj = qp.IBIS(dites=offset,
                            dites_err=offset_err,
                            ibias=tes_bias,
                            ibias_err=np.zeros_like(tes_bias),
                            rsh=param_dict['rshunt'],
                            rsh_err=param_dict['rshunt_err'],
                            rp_guess=rp,
                            rp_err_guess=rp_err,
                            chan_names=labels,
                            fitsc=fitsc,
                            normalinds=list(range_normal),
                            scinds=list(range_sc),)
                
            ivobj.analyze(**kwargs)
            
            rn_check = ivobj.rnorm[0, 0]
            if (lgc_invert_offset == 'auto' and rn_check<0):
                lgc_invert_offset = True
                ivobj.dites = -1*ivobj.dites
                ivobj.analyze(**kwargs)
            
                     
            # store results in dataframe/series
            df['lgc_invert_offset'] = lgc_invert_offset
            df['vb'] = ivobj.vb[0,0,:]
            df['vb_err'] = ivobj.vb_err[0,0,:]
            
            for itype, data_type in enumerate(data_types):
                
                # parameter dependend of bias point
                df['p0_' + data_type] = ivobj.ptes[0, itype]
                df['p0_err_' + data_type] = ivobj.ptes_err[0, itype]
                df['r0_' + data_type] = ivobj.r0[0, itype]
                df['r0_err_' + data_type] = ivobj.r0_err[0, itype]
                df['i0_' + data_type] = ivobj.ites[0, itype]
                df['i0_err_' + data_type] = ivobj.ites_err[0, itype]

                # get results independent of sweep
                rp = ivobj.rp[0, itype]
                rp_err =  ivobj.rp_err[0, itype]
                rn = ivobj.rnorm[0, itype]
                rn_err = ivobj.rnorm_err[0, itype]
                i0_off = ivobj.ioff[0, itype]
                i0_off_err = ivobj.ioff_err[0, itype]
                ibias_off = ivobj.ibias_off[0, itype]
                ibias_off_err = ivobj.ibias_off_err[0, itype]
                ibias_true = ivobj.ibias_true[0, itype]
                ibias_true_err = ivobj.ibias_true_err[0, itype]


                # store in df and dictionary
                df['rp_' + data_type] = rp
                df['rp_err_' + data_type] = rp_err
                df['rn_' + data_type] = rn
                df['rn_err_' + data_type] = rn_err
                df['i0_off_' + data_type] = i0_off
                df['i0_off_err_' + data_type] = i0_off_err
                df['ibias_off_' + data_type] = ibias_off
                df['ibias_off_err_' + data_type] = ibias_off_err
                df['ibias_true_' + data_type] = ibias_true
                df['ibias_true_err_' + data_type] = ibias_true_err
                df['percent_rn_' + data_type] = round(1000*df['r0_' + data_type]/rn)/10
                

                # store also in in pandas series
                results = dict()
                results['rp'] = rp
                results['rp_err'] = rp_err
                results['rn'] = rn
                results['rn_err'] = rn_err
                results['rshunt'] = param_dict['rshunt']
                results['rshunt_err'] = param_dict['rshunt_err']
                results['i0_off'] = i0_off
                results['i0_off_err'] = i0_off_err
                results['ibias_off'] = ibias_off
                results['ibias_off_err'] = ibias_off_err
                results['lgc_invert_offset'] = lgc_invert_offset
                results['close_loop_norm'] = (
                    df['close_loop_norm_' + data_type].values[0]
                )
                results['output_variable_offset'] = (
                    df['output_variable_offset_' + data_type].values[0]
                )
                results['output_variable_gain'] = (
                    df['output_variable_gain_' + data_type].values[0]
                )
                results['group_name_sweep'] = (
                    df['group_name_' + data_type].values[0]
                )


                # calculate i0 variable offset
                norm = results['close_loop_norm'] 
                voltage_offset = results['output_variable_offset']
                gain =  results['output_variable_gain']
                results['i0_variable_offset'] = (
                    voltage_offset * gain/norm
                )
                
                # convert to pandas series
                series = pd.Series(results)
                lgc_didv_data = False
                if data_type == 'didv':
                    lgc_didv_data = True
                self.set_ivsweep_result(chan,
                                        series,
                                        lgc_didv_data=lgc_didv_data,
                                        metadata=None,
                                        tag=tag)
                            
            # save data
            self._iv_objects[chan] = ivobj
            self.set_ivsweep_data(chan, df, tag=tag)
            
            if lgc_plot:
                print(f'Channel {chan} IV/dIdV Sweep Fits:')
                ivobj.plot_all_curves(lgcsave=False,
                                      savepath=None,
                                      savename=None)
