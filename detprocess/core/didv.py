import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io
import qetpy as qp
from glob import glob
import vaex as vx
from pathlib import Path
from scipy.signal import unit_impulse
from detprocess.core.filterdata import FilterData


class DIDVAnalysis(FilterData):
    """
    Class to manage didv calculations using 
    QETpy. DIDV data are stored 
    """

    def __init__(self, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information

        """

        # instantiate base class
        super().__init__(verbose=verbose)
        
        # initialize didv objects
        self._didv_data = None
        

    def clear(self, channels=None):
        """
        Clear all data
        """
        
        # clear data
        if channels is None:
            self._didv_data = None
        else:
            if isinstance(channels, str):
                channels = [channels]
            for chan in channels:
                if chan in self._didv_data.keys():
                    self._didv_data.pop(chan)


    def get_didv_data(self, channel):
        """
        Get dictionary with didv data
        """

        if channel in self._didv_data.keys():
            return self._didv_data[channel]
        else:
            raise ValueError(f'ERROR: No didv data available '
                             f'for channel "{channel}"!')

    
    def get_dpdi(self, channel, poles=3):
        """
        Get dpdi data and frequencies
        """

        
        if channel not in self._didv_data.keys():
            raise ValueError(f'ERROR: No didv data available '
                             f'for channel "{channel}"!')
        
        list_of_fitted_poles = (
            self._didv_data[channel]['didvobj'].get_list_fitted_poles()
        )
        
        if poles not in list_of_fitted_poles:
            raise ValueError(f'ERROR: No {poles}-poles fit found for '
                             f'channel {channel}!')

            
        # check dpdi available
        poles_str = str(poles) + 'poles'
        if 'dpdi_' + poles_str not in self._didv_data[channel]:
            raise ValueError(f'ERROR: No dpdi found for '
                             'channel {channel}!')
     
        dpdi = self._didv_data[channel]['dpdi_' +  poles_str]
        dpdi_freqs = self._didv_data[channel]['dpdi_freqs_' +  poles_str]

        return dpdi_freqs, dpdi
        
        
    def get_qetpy_object(self, channel):
        """
        Get QETpy object
        """
        
        didv_data = self.get_didv_data(channel)
        if 'didvobj' in didv_data.keys():
            return  didv_data['didvobj']
        else:
            raise ValueError(f'ERROR: No dIdV QETpy object available '
                             f'for channel "{channel}"!')

    def get_bias_params(self, channel, poles=3):
        """
        Get "bias" (I0, R0, P0) parameters
        (can be either "true" current or set by user)
        """
        
        fit_result = self.get_fit_results(channel, poles=poles)
        if 'biasparams' in fit_result.keys():
            return fit_result['biasparams']
        else:
            raise ValueError(f'ERROR: No bias parameters available '
                             f'for channel "{channel}"!')


    def get_bias_params_infinite_loop_gain(self, channel, poles=3):
        """
        Get "bias" (I0, R0, P0) parameters
        (can be either "true" current or set by user)
        """
        
        fit_result = self.get_fit_results(channel, poles=poles)
        if 'biasparams_infinite_lgain' in fit_result.keys():
            return fit_result['biasparams_infinite_lgain']
        else:
            raise ValueError(f'ERROR: No bias parameters available '
                             f'for channel "{channel}"!')


        
    def process_raw_data(self, channels,
                         raw_path, series=None,
                         overwrite=False):
        
        """
        Set dIdV data for a specific channel
        form raw traces. Apply cuts and pre-process
        data. The function can be called multiple
        times for different channels 
        """

        # loop channel and process data
        if isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            self._process_raw_data(chan, raw_path,
                                   series=series,
                                   overwrite=overwrite)

            
        
    def set_processed_data(self, channels,
                           data,
                           overwrite=False):
        
        """
        Set dIdV data from already processed raw 
        data, in the form of pandas series or 
        dictionary. The function can be called multiple
        times for different channels 
        """


        didv_data = data.copy()
        
        # check channels and data
        if isinstance(channels, str):
            if channels not in data.keys():
                didv_data = dict()
                didv_data[channels] = data     
            channels = [channels]
        
        for chan in channels:
            if chan not in didv_data.keys():
                raise ValueError(f'ERROR: Unable to find channel '
                                 f' {chan} in data dictionary!')

        # set data
        for chan in channels:
            self._set_processed_data(chan,
                                     didv_data[chan],
                                     overwrite=overwrite)

    def set_ivsweep_results_from_data(self,
                                      channel,
                                      results,
                                      iv_type='noise'):
        """
        Set results from IV sweeps, in particular
        I0 offset, V0 offset
        """

        # check
        if isinstance(results, pd.Series):
            results = results.to_dict()
        elif not isinstance(results, dict):
            raise ValueError(
                'ERROR: "results" should be a dictionary '
                ' or pandas series')
        
        # check channel
        if not isinstance(channel, str):
            raise ValueError(f'ERROR: This function is for single channel!' )
        
        if channel not in self._didv_data.keys():
            raise ValueError(f'ERROR: Channel {channel} not found. '
                             ' You first need to set data!' )
    

        # save
        pd_series = pd.Series(results.copy())
        self.set_ivsweep_results(channel,
                                 pd_series,
                                 iv_type)
                    
        # keep a copy as part of didv data
        self._didv_data[channel]['ivsweep_results'] = results.copy()
            
        # check if Rp, Rshunt, R0 available
        if 'rp' in results.keys():
            rp = results['rp']
        else:
            raise ValueError(f'ERROR: "rp" is missing from IV results for '
                             f'channel {channel}. It needs to be added!')
            
        r0 = None
        if 'r0' in results.keys():
            r0 = results['r0']
            
        rshunt = None
        if 'rshunt' in results.keys():
            rshunt = results['rshunt']
                      
        # replace internal object data
        if rp is not None:
            self._didv_data[channel]['data_config']['rp'] = rp
            self._didv_data[channel]['didvobj']._rp = rp
            
        if r0 is not None:
            self._didv_data[channel]['didvobj']._r0 = r0
            
        if (rshunt is not None
            and self._didv_data[channel]['data_config']['rshunt'] is None):
            self._didv_data[channel]['data_config']['rshunt'] = rshunt
            self._didv_data[channel]['didvobj']._rsh = rshunt
        

    def set_ivsweep_results_from_file(self,
                                      channels,
                                      file_name,
                                      iv_type='noise',
                                      include_bias_parameters=False):
        """
        Set results from IV sweeps, in particular
        I0 offset, V0 offset
        """
        
        # check channels
        if isinstance(channels, str):
            channels = [channels]
           
        for chan in channels:
            if chan not in self._didv_data.keys():
                raise ValueError(f'ERROR: Channel {channel} not found. '
                                 ' You first need to set data!' )
           
     
        # load data from file
        if file_name is not None:
            self.load_hdf5(file_name)

            
        # loop channels
        for chan in channels:

            #  get tes bias
            tes_bias = None
            if include_bias_parameters:
                data_config = self._didv_data[chan]['data_config']
                tes_bias = data_config['tes_bias']

            # get results
            results =  self.get_ivsweep_results(
                chan,
                iv_type=iv_type,
                include_bias_parameters=include_bias_parameters,
                tes_bias=tes_bias,
            )

            # save in include bias parameters
            if include_bias_parameters:
                pd_series = pd.Series(results.copy())
                self.set_ivsweep_results(chan,
                                         pd_series,
                                         iv_type)
                
            # keep a copy as part of didv data
            self._didv_data[chan]['ivsweep_results'] = results.copy()
            
            # check if Rp, Rshunt, R0 available
            if 'rp' in results.keys():
                rp = results['rp']
            else:
                raise ValueError(f'ERROR: "rp" is missing from IV results for '
                                 f'channel {chan}. It needs to be added!')
            
            r0 = None
            if 'r0' in results.keys():
                r0 = results['r0']
            
            rshunt = None
            if 'rshunt' in results.keys():
                rshunt = results['rshunt']
                      
            # replace internal object data
            if rp is not None:
                self._didv_data[chan]['data_config']['rp'] = rp
                self._didv_data[chan]['didvobj']._rp = rp
            
            if r0 is not None:
                self._didv_data[chan]['didvobj']._r0 = r0
                
            if (rshunt is not None
                and self._didv_data[chan]['data_config']['rshunt'] is None):
                self._didv_data[chan]['data_config']['rshunt'] = rshunt
                self._didv_data[chan]['didvobj']._rsh = rshunt
        
            
        
    
    def dofit(self, list_of_poles, channels=None,
              fcutoff=np.inf, bounds=None, guess_params=None,
              guess_isloopgainsub1=None, lgc_fix=None,
              add180phase=False, dt0=1.5e-6,
              lgc_plot=False,
              tag='default'):
        
        """
        Do dIdV fit for one or more channels. The function
        "set_data" needs to be called prior "dofit" to store
        didv data
        
        """

        # check if data available for each channel
        if channels is None:
            channels = self._didv_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

            
        for chan in channels:
            if chan not in self._didv_data.keys():
                print(f'ERROR: dIdV data for channel {chan} '
                      'not available. Set data first!')
                return
            

        # poles
        if isinstance(list_of_poles, int):
            list_of_poles = [list_of_poles]
        list_of_poles.sort()
                   
            
        # loop channel and do fit
        for chan in channels:
            
            # get didv obj
            didvobj = self._didv_data[chan]['didvobj']
            didvobj._add180phase = add180phase
            didvobj._dt0 = dt0
            
            for poles in list_of_poles:
                didvobj.dofit(
                    poles,
                    fcutoff=fcutoff,
                    bounds=bounds,
                    guess_params=guess_params,
                    guess_isloopgainsub1=guess_isloopgainsub1,
                    lgcfix=lgc_fix
                )
            
            # replace object with results included
            self._didv_data[chan]['didvobj'] =  didvobj


            if lgc_plot:
                
                if 3 in list_of_poles:
                    self.print_fit_result(chan, poles=3)
                elif 2 in list_of_poles:
                    self.print_fit_result(chan, poles=2)
                
                self.plot_fit_result(chan)


            
    
    
    def calc_smallsignal_params(self,
                                channels=None,
                                calc_true_current=True,
                                inf_loop_gain_approx='auto',
                                priors_fit_method=False,
                                lgc_diagnostics=False):
        """
        Calculate small signal parameters with uncertainties
        (optionally) using current from IV sweep.
        
        """

        # check channels
        if channels is None:
            channels = self._didv_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            if chan not in self._didv_data.keys():
                raise ValueError(f'ERROR: dIdV fit not available for '
                                 f'channel {chan}! Use "dofit" first.')
       
        # loop channel
        for chan in channels:

            if self._verbose:
                print(f'INFO: Calculating small signal parameters '
                      f'uncertainties for channel {chan}')
            
            # get didvobj
            didvobj = self._didv_data[chan]['didvobj']
            
            # IV sweep result
            ivsweep_results = self._didv_data[chan]['ivsweep_results']
            if ivsweep_results is None:
                raise ValueError(f'ERROR: No IV sweep result for channel '
                                 f'{chan}. first set results using '
                                 '"set_ivsweep_results" function!')

            # check ivsweep results for case no true I0/R0 calculation
            if not calc_true_current:
                extra_pars = ['i0','i0_err', 'r0', 'r0_err']
                for par in extra_pars:
                    if par not in ivsweep_results:
                        raise ValueError(
                            'ERROR: parameters "i0", "i0_err", '
                            '"r0", "r0_err" from IV sweep required if '
                            'true current not calculated!'
                        )
            
            # data config
            data_config = self._didv_data[chan]['data_config']
            tes_bias = data_config['tes_bias']
            output_variable_offset = data_config['output_variable_offset']
            close_loop_norm = data_config['close_loop_norm']
            output_variable_gain = data_config['output_variable_gain']
            

            # calc small signal parameters
            didvobj.calc_smallsignal_params(
                ivsweep_results,
                calc_true_current=calc_true_current,
                tes_bias=tes_bias,
                close_loop_norm=close_loop_norm,
                output_variable_gain=output_variable_gain,
                output_variable_offset=output_variable_offset,
                inf_loop_gain_approx=inf_loop_gain_approx,
                lgc_diagnostics=lgc_diagnostics)

            # replace
            self._didv_data[chan]['didvobj'] = didvobj
      
    def calc_bias_params_infinite_loop_gain(self, channels=None):
                                           
        """
        Calculate I0,R0, and P0 with infinite loop gain
        approximation
        """
                    
        # check channels
        if channels is None:
            channels = self._didv_data.keys()
        elif isinstance(channels, str):
            channels = [channels]
        nb_channels = len(channels)


        for chan in channels:
            if chan not in self._didv_data.keys():
                raise ValueError(f'ERROR: channel {chan} not available!')
            didvobj = self._didv_data[chan]['didvobj']
            list_of_fitted_poles =  didvobj.get_list_fitted_poles()
            if (list_of_fitted_poles is None
                or (2 not in list_of_fitted_poles
                    and 3 not in list_of_fitted_poles)):
                raise ValueError(f'ERROR: No fit available for '
                                 f'channel {chan}!')
            
        # loop channel
        for chan in channels:

            if self._verbose:
                print(f'INFO: Calculating bias parameters with '
                      f'infinite loop gain approximation '
                      f'for channel {chan}')
            # get data config
            data_config = self._didv_data[chan]['data_config']

                
            # get didvobj
            didvobj = self._didv_data[chan]['didvobj']
            list_of_fitted_poles =  didvobj.get_list_fitted_poles()
         
            # loop poles
            for poles in list_of_fitted_poles:
                
                if (poles != 2 and poles != 3):
                    continue
                
                fitresult = didvobj.fitresult(poles)

                ibias = None
                ibias_err = 0
                rp = None
                if ('biasparams' in fitresult.keys() and
                    fitresult['biasparams'] is not None):
                 
                    ibias = fitresult['biasparams']['ibias']
                    if 'ibias_err' in  fitresult['biasparams']:
                        ibias_err =  fitresult['biasparams']['ibias_err']
                    else:
                        ibias_err = 0
                    rp = fitresult['biasparams']['rp']
                else:
                    ibias = data_config['tes_bias']
                    if 'rp' not in data_config:
                        raise ValueError('ERROR: Unable to find rp!'
                                         'use "set_ivsweep_results()"  first')
                    rp =  data_config['rp']

                didvobj.calc_bias_params_infinite_loop_gain(
                    poles,
                    tes_bias=ibias,
                    tes_bias_err=ibias_err,
                    rp=rp)

                self._didv_data[chan]['didvobj'] = didvobj
               
                                                          
            
   
    def calc_dpdi(self, freqs, channels=None, list_of_poles=None,
                  lgc_plot=False):
        """
        calculate dpdi
        """

        # check channels
        if channels is None:
            channels = self._didv_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            if chan not in self._didv_data.keys():
                raise ValueError(f'ERROR: No data found for '
                                 'channel {chan}!')
            
        for chan in channels:
            
            # get didvobj
            didvobj = self._didv_data[chan]['didvobj']

            # list of poles
            list_of_fitted_poles =  didvobj.get_list_fitted_poles()
            if list_of_poles is None:
                list_of_poles = list_of_fitted_poles
            elif isinstance(list_of_poles, int):
                list_of_poles = [list_of_poles]


            # loop poles
            for poles in list_of_poles:

                if poles not in list_of_poles:
                    raise ValueError(f'ERROR: No fit available for poles '
                                     f'{poles}. Use "dofit" function first!')

                fitresult = didvobj.fitresult(poles)
            
                dpdi, dpdi_err = qp.get_dPdI_with_uncertainties(
                    freqs, fitresult,
                    lgcplot=lgc_plot)
                
                poles_str = str(poles) + 'poles'
                self._didv_data[chan]['dpdi_' +  poles_str] = dpdi
                self._didv_data[chan]['dpdi_err_' +  poles_str] = dpdi_err
                self._didv_data[chan]['dpdi_freqs_' + poles_str] = freqs

                
                
    def calc_energy_resolution(self, channel, psd,
                               poles=3, fs=None, template=None,
                               collection_eff=1,
                               lgc_power_template=False):
        """
        Get energy resolution based using  calculated dpdi and input
        psd and template (dirac delta power input if template is None)
        Default is template in current. 
        """

        # check data availability
        if channel not in self._didv_data.keys():
            raise ValueError(f'ERROR: No data found for '
                             'channel {channel}!')

        # check if poles fitted
        if poles != 3:
            raise ValueError(f'ERROR: Resoultion can only be calculated for '
                             '3-poles fit currently!')
        list_of_fitted_poles = (
            self._didv_data[channel]['didvobj'].get_list_fitted_poles()
        )
        if poles not in list_of_fitted_poles:
            raise ValueError(f'ERROR: No {poles}-poles fit found for '
                             'channel {channel}!')

            
        # check dpdi available
        poles_str = str(poles) + 'poles'
        if 'dpdi_' + poles_str not in self._didv_data[channel]:
            raise ValueError(f'ERROR: No dpdi found for '
                             'channel {channel}!')
     
        dpdi = self._didv_data[channel]['dpdi_' +  poles_str]
        dpdi_freqs = self._didv_data[channel]['dpdi_freqs_' +  poles_str]

        # sample rate
        if fs is None:

            if 'fs' not in self._didv_data[channel]['data_config']:
                raise ValueError(f'ERROR: sample rate (fs) required!')

            fs = self._didv_data[channel]['data_config'][fs]
        

        if template is None:

            # dirac delta input power
            nbins = psd.shape[-1]
            pretrigger_index = nbins//2
            template = unit_impulse(nbins, idx=pretrigger_index)

        elif not lgc_power_template:
            template = qp.convert_template_to_power(template,  dpdi=dpdi)

            
        # calculate energy resolution
        resolution = qp.utils.energy_resolution(
            psd, template, dpdi, fs,
            collection_eff=collection_eff)

        # store
        res_dict = dict()
        res_dict['psd'] = psd
        res_dict['template'] = template
        res_dict['fs'] = fs
        res_dict['collection_efficiency'] = collection_eff
        res_dict['energy_resolution'] = resolution
        self._didv_data[channel]['resolution'] = res_dict
        
        return resolution
            
                
    def dofit_prior(self, channels=None, list_of_poles=None,
                    use_ivsweep_current=True):
        """
        Do prior fit to calculate uncertainties
        """

        # check channels
        if channels is None:
            channels = self._didv_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            if chan not in self._didv_data.keys():
                raise ValueError(f'ERROR: dIdV fit result not found for '
                                 'channel {chan}! Use "dofit" first.')
            
        
        # loop channel
        for chan in channels:

            if self._verbose:
                print(f'INFO: Do prior fit(s) for '
                      ' channel {channel}')

            # get didvobj
            didvobj = self._didv_data[chan]['didvobj']

            # instanciate priors

            didvobj_prior = qp.DIDVPriors(
                rawtraces=None,
                fs=didvobj._fs,
                sgfreq=didvobj._sgfreq,
                sgamp=didvobj._sgamp,
                rsh=didvobj._rsh,
                dutycycle=didvobj._dutycycle
            )
                
            didvobj_prior._time = didvobj._time
            didvobj_prior._freq = didvobj._freq
            didvobj_prior._didvmean = didvobj._didvmean
            didvobj_prior._didvstd = didvobj._didvstd
            didvobj_prior._offset = didvobj._offset
            didvobj_prior._offset_err = didvobj._offset_err
            didvobj_prior._tmean = didvobj._tmean
            didvobj_prior._dt0 = didvobj._dt0


            # loop poles
            for poles_select in poles:
                
                if poles_select == 2:
                    
                    results = didvobj.fitresult(2)

                    # prior array
                    priors = np.ones(8)
                    priors[0] = results['smallsignalparams']['rsh']
                    priors[1] = results['smallsignalparams']['rp']
                    priors[2] = results['smallsignalparams']['r0']
                    priors[3] = results['smallsignalparams']['beta']
                    priors[4] = results['smallsignalparams']['l']
                    priors[5] = results['smallsignalparams']['L']
                    priors[6] = results['smallsignalparams']['tau0']
                    priors[7] = results['smallsignalparams']['dt']
                    

                    # prior covariance
                    priors_cov = np.zeros((8,8))
                        
                    rp_sig = row.rp_err
                    rshunt_sig = self.rshunt_err
                    r0_sig = row.r0_err
                    
                    for ii in range(len(priors_cov)):
                        priors_cov[ii,ii] = (priors[ii]*.1)**2
                       
                    priors_cov[0,0] = rshunt_sig**2
                    priors_cov[1,1] = rp_sig**2
                    priors_cov[2,2] = r0_sig**2
                    priors_cov[0,1] = priors_cov[1,0] = .5*rshunt_sig*rp_sig
                    priors_cov[0,2] = priors_cov[2,0] = .5*rshunt_sig*r0_sig
                    priors_cov[1,2] = priors_cov[2,1] = -.2*rp_sig*r0_sig
                    
                    didvobj_prior.dofit(poles=2,
                                        priors=priors,
                                        priorscov=priors_cov)
                    

                elif poles_select == 3:

                    results = didvobj.fitresult(3)
                    
                    # priors array
                    priors = np.ones(10)
                    priors[0] = result['smallsignalparams']['rsh']
                    priors[1] = result['smallsignalparams']['rp']
                    priors[2] = result['smallsignalparams']['r0']
                    priors[3] = result['smallsignalparams']['beta']
                    priors[4] = result['smallsignalparams']['l']
                    priors[5] = result['smallsignalparams']['L']
                    priors[6] = result['smallsignalparams']['tau0']
                    priors[7] = result['smallsignalparams']['gratio']
                    priors[8] = result['smallsignalparams']['tau3']
                    priors[9] = result['smallsignalparams']['dt']
                    
                    
                    # priors covariance
                    priors_cov = np.zeros((10,10))
                    
                    for ii in range(len(priors_cov)):
                        priors_cov[ii,ii] = (priors[ii]*.1)**2
                        
                    r0_sig3 = priors[2]*.15
                    priors_cov[0,0] = rshunt_sig**2
                    priors_cov[1,1] = rp_sig**2
                    priors_cov[2,2] = r0_sig3**2
                    priors_cov[0,1] = priors_cov[1,0] = .5*rshunt_sig*rp_sig
                    priors_cov[0,2] = priors_cov[2,0] = .5*rshunt_sig*r0_sig3
                    priors_cov[1,2] = priors_cov[2,1] = -.2*rp_sig*r0_sig3

                    didvobj_prior.dofit(poles=3,
                                        priors=priors,
                                        priorscov=priors_cov)
                        
            # save
            self._didv_data[chan]['didvobj_prior'] = didvobj_prior
              
    def get_fit_results(self, channel, poles):
        """
        Get fit result
        """

        # check if object available
        if channel not in self._didv_data.keys():
            raise ValueError(f'ERROR: dIdV data not available for '
                             f'channel {chan}!')
        
        # check if fit done
        result = self._didv_data[channel]['didvobj'].fitresult(poles)
        if not result:
            if self._verbose:
                print(f'WARNING: {channel}: No fit result found for poles {poles}! '
                      'Returning empty dictionary.')
        return result

    
    def plot_fit_result(self, channels=None,
                        lgc_plot_fft=True, lgc_gray_mean=True,
                        lgc_didv_freq_filt=True,
                        zoom_factor=None, fcutoff=2e4,
                        lgc_save=False, save_path=None,
                        save_name=None):
        """
        Plot fit results for multiple channels
        """

        
        # check channels
        if channels is None:
            channels = self._didv_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            if chan not in self._didv_data.keys():
                raise ValueError(f'ERROR: dIdV fit result not found for '
                                 'channel {chan}! Use "dofit" first.')
                  
        # look chanels
        for chan in channels:
            
            didvobj = self._didv_data[chan]['didvobj']

            # check if any results
            is_fitted = False
            poles_list = [1,2,3]
            for poles in poles_list:
                if didvobj.fitresult(poles):
                    is_fitted = True
            if not is_fitted:
                print(f'WARNING: No fit result available for '
                      'channel {chan}!')
                continue
                    
            # display
            if self._verbose:
                print(f'\n{chan} dIdV Fit Plots:')

            didvobj.plot_full_trace(
                didv_freq_filt=lgc_didv_freq_filt,
                gray_mean=lgc_gray_mean,
                saveplot=lgc_save,
                savepath=save_path,
                savename=save_name,
                lp_cutoff=fcutoff,
            )

            if zoom_factor is not None:
                didvobj.plot_zoomed_in_trace(
                    didv_freq_filt=lgc_didv_freq_filt,
                    saveplot=lgc_save,
                    savepath=save_path,
                    savename=save_name,
                    lp_cutoff=fcutoff,
                    zoomfactor=zoom_factor,
                ) 

            if lgc_plot_fft:
                didvobj.plot_re_im_didv(
                    poles='all',
                    saveplot=lgc_save,
                    savepath=save_path,
                    savename=save_name,
                )

                
    def print_fit_result(self, channels=None, poles=3):
        """
        Print fit information
        """

        # check channels
        if channels is None:
            channels = self._didv_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

        for chan in channels:
            if chan not in self._didv_data.keys():
                raise ValueError(f'ERROR: dIdV fit result not found for '
                                 'channel {chan}! Use "dofit" first.')
                  
        # look chanels
        for chan in channels:
            
            didvobj = self._didv_data[chan]['didvobj']

            # check if any results
            results = didvobj.fitresult(poles)
            if not results:
                print(f'WARNING: No fit result available for '
                      'channel {chan}!')
                continue
            
            print(f'\n{chan} dIdV Fit Result:')

            
            print('Fit chi2/Ndof = {:.3f}'.format(results['cost']))
            print('\nFit time constants, NOT dIdV Poles: ')
            print('Tau1: {:.3g} s'.format(np.abs(results['params']['tau1'])))
            print('Tau2: {:.3g} s'.format(results['params']['tau2']))
            print('Tau3: {:.4g} s'.format(results['params']['tau3']))
            print(' ')
            print('\nTrue dIdV Poles: ')
            print('Tau_plus: {:.3g} s'.format(results['falltimes'][0]))
            print('Tau_minus: {:.3g} s'.format(results['falltimes'][1]))
            print('Tau_third: {:.4g} s'.format(results['falltimes'][2]))
            
            if 'ssp_light' in results:
                vals_vector = results['ssp_light']['vals']
                sigmas_vector = results['ssp_light']['sigmas']
                print('\nSmall Signal Parameters:')
                print('l (loop gain) = {:.3f} +/- {:.4f}'.format(
                    vals_vector['l'], sigmas_vector['sigma_l']))
                print('beta = {:.3f} +/- {:.4f}'.format(
                    vals_vector['beta'], sigmas_vector['sigma_beta']))
                print('gratio = {:.3f} +/- {:.4f}'.format(
                    vals_vector['gratio'], sigmas_vector['sigma_gratio']))
                print('tau0 = {:.3g} +/- {:.4g} ms'.format(
                    vals_vector['tau0']*1e3, sigmas_vector['sigma_tau0']*1e3))
                print('L = {:.3f} +/- {:.4f} nH'.format(
                    vals_vector['L']*1e9, sigmas_vector['sigma_L']*1e9))
                
        



                
    def compare_with_ivsweep(self, channel, poles=3):
        """
        Create a dataframe to compare results with
        IV sweep
        """

        # check channel
        if channel not in self._didv_data.keys():
            raise ValueError(f'ERROR:  channel {channel} not found!')
                 

        didv_results = self.get_fit_results(channel, poles=poles)
        ivsweep_results = self.get_ivsweep_results(channel)

        data = dict()
        
        
        if ('biasparams' in didv_results
            and didv_results['biasparams'] is not None):

            # define parameter list
            param_list = ['r0','i0','p0']
            norm_list = [1e3, 1e6, 1e15]
            label_list = ['R0 [mOhms]','I0 [muAmps]','P0 [fWatts]']
            label_inf_lgain_list = ['R0 Inf loop gain [mOhms]',
                                    'I0 Inf loop gain [muAmps]',
                                    'P0 Inf loop gain [fWatts]']

            # results
            results = didv_results['biasparams']
            results_infinite_lgain = None
            if 'biasparams_infinite_lgain' in didv_results:
                results_infinite_lgain = (
                    didv_results['biasparams_infinite_lgain']
                )

            # loop parameters
            for iparam, param in enumerate(param_list):
                
                if param not in ivsweep_results:
                    continue

                # normalization/label
                norm = norm_list[iparam]
                label = label_list[iparam]
                label_infinite_lgain = label_inf_lgain_list[iparam]

                
                # build string dIdV
                val = results[param]
                val_err = results[param + '_err']
                val_str = '{:.3g} +/- {:.4g}'.format(
                    val*norm, val_err*norm
                )

                
                # build string IV 
                iv_val = ivsweep_results[param]
                iv_val_err = ivsweep_results[param + '_err']
                iv_val_str = '{:.3g} +/- {:.4g}'.format(
                    iv_val*norm, iv_val_err*norm
                )
                    
                data[label] = [val_str, iv_val_str]

                    
                # infinite loop gain
                if (results_infinite_lgain is not None
                    and 'r0_infinite_lgain' in ivsweep_results):

                    # build string dIdV
                    val = results_infinite_lgain[param]
                    val_err = results_infinite_lgain[param + '_err']
                    val_str = '{:.3g} +/- {:.4g}'.format(
                        val*norm, val_err*norm
                    )

                    # build string IV 
                    iv_val = ivsweep_results[param + '_infinite_lgain']
                    iv_val_err = ivsweep_results[param + '_err_infinite_lgain']
                    iv_val_str = '{:.3g} +/- {:.4g}'.format(
                        iv_val*norm, iv_val_err*norm
                    )
                    
                    data[label_infinite_lgain] = [val_str, iv_val_str]


        # small signal parmaters
        if ('ssp_light' in didv_results
            and 'didv_fit_l' in ivsweep_results):
            
            # define parameter list
            param_list = ['l','beta','gratio', 'L', 'tau0']
            norm_list = [1, 1, 1, 1e9, 1e3]
            label_list = ['l (loop gain)','beta', 'gratio',
                          'L [nH]', 'tau0 [ms]']

            
            vals_vector = didv_results['ssp_light']['vals']
            sigmas_vector = didv_results['ssp_light']['sigmas']


            # loop parameters
            for iparam, param in enumerate(param_list):

                param_iv = 'didv_fit_' + param
                
                # normalization/label
                norm = norm_list[iparam]
                label = label_list[iparam]
             
                
                # build string dIdV
                val = vals_vector[param]
                val_err = sigmas_vector['sigma_' + param]
                val_str = '{:.3g} +/- {:.4g}'.format(
                    val*norm, val_err*norm
                )
                
                # build string IV 
                iv_val = ivsweep_results[param_iv]
                iv_val_err = ivsweep_results[param_iv + '_err']
                iv_val_str = '{:.3g} +/- {:.4g}'.format(
                    iv_val*norm, iv_val_err*norm
                )

                # store data
                data[label] = [val_str, iv_val_str]

        #  fall times
        if ('falltimes' in didv_results
            and 'didv_fit_tau+' in ivsweep_results):
            
            # define parameter list
            param_list = ['tau+','tau-','tau3']
            norm_list = [1e6, 1e6, 1e6]
            label_list = ['tau+ [usec]', 'tau- [usec]', 'tau3 [usec]']


            falltimes = didv_results['falltimes']      
        

            # loop parameters
            for iparam, param in enumerate(param_list):

                if (poles == 2 and iparam>1):
                    break

                
                param_iv = 'didv_fit_' + param
                
                # normalization/label
                norm = norm_list[iparam]
                label = label_list[iparam]
             
                
                # build string dIdV
                val = falltimes[iparam]
                val_str = '{:.3g}'.format(val*norm)
                
                
                # build string IV 
                iv_val = ivsweep_results[param_iv]
                iv_val_str = '{:.3g}'.format(iv_val*norm)

                # save
                data[label] = [val_str, iv_val_str]

                  
                        
        df = None
        if data:
            df = pd.DataFrame.from_dict(data, orient='index', columns=['dIdV', 'IV Sweep'])
            
        return df


                        
            
    def _get_file_list(self, file_path,
                       series=None):
        """
        Get file list from path. Return as a dictionary
        with key=series and value=list of files

        Parameters
        ----------

        file_path : str or list of str 
           raw data group directory OR full path to HDF5  file 
           (or list of files). Only a single raw data group 
           allowed 
        
        series : str or list of str, optional
            series to be process, disregard other data from raw_path


        Return
        -------
        
        file_list : list of files

        base_path :  str
           base path of the raw data

        group_name : str
           group name of raw data

        """

        # convert file_path to list 
        if isinstance(file_path, str):
            file_path = [file_path]
            
            
        # initialize
        file_list = list()
        base_path = None
        group_name = None


        # loop files 
        for a_path in file_path:
                   
            # case path is a directory
            if os.path.isdir(a_path):

                if base_path is None:
                    base_path = str(Path(a_path).parent)
                    group_name = str(Path(a_path).name)
                            
                if series is not None:
                    if series == 'even' or series == 'odd':
                        file_name_wildcard = series + '_*.hdf5'
                        file_list = glob(a_path + '/' + file_name_wildcard)
                    else:
                        if not isinstance(series, list):
                            series = [series]
                        for it_series in series:
                            file_name_wildcard = '*' + it_series + '_*.hdf5'
                            file_list.extend(
                                glob(a_path + '/' + file_name_wildcard)
                            )
                else:
                    file_list = glob(a_path + '/*.hdf5')
               
                # check a single directory
                if len(file_path) != 1:
                    raise ValueError('Only single directory allowed! ' +
                                     'No combination files and directories')
                
                    
            # case file
            elif os.path.isfile(a_path):

                if base_path is None:
                    base_path = str(Path(a_path).parents[1])
                    group_name = str(Path(Path(a_path).parent).name)
                    
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
                raise ValueError('File or directory "' + a_path
                                 + '" does not exist!')
            
        if not file_list:
            raise ValueError('ERROR: No raw input data found. Check arguments!')

        # sort
        file_list.sort()

      
        return file_list, base_path, group_name


    
    def _process_raw_data(self, channel,
                          raw_path, series=None,
                          overwrite=False):
        
        """
        Set dIdV data for a specific channel
        form raw traces. Apply cuts and pre-process
        data. The function can be called multiple
        times for different channels 
        """

        # check if data can be overwritte
        if  self._didv_data is None:
             self._didv_data = dict()
        else:
            if (not overwrite and
                channel in self._didv_data.keys()):
                print(f'ERROR: Data for channel {channel} '
                      'already available. Set overwrite=True')
                return
        
        # get file list 
        file_list, base_path, group_name = self._get_file_list(raw_path, series)
        if file_list is None:
            raise ValueError(
                'ERROR: No didv raw data found!'
            )

          
        # Instantiate QETpy DIDV Object
        didvobj = None
        if self._verbose:
            print(f'INFO: Getting raw data and processing channel {channel}')
            
        # instantiate H5 reader
        h5 = h5io.H5Reader()
        
        # get traces
        traces, info = h5.read_many_events(
            filepath=file_list,
            detector_chans=channel,
            output_format=2, 
            include_metadata=True,
            adctoamp=True)
        
        traces = traces[:,0,:]
        fs = info[0]['sample_rate']
            
        # get detector settings
        detector_settings = h5.get_detector_config(file_name=file_list[0])

        tes_bias = float(detector_settings[channel]['tes_bias'])
        output_gain = float(detector_settings[channel]['output_gain'])
        close_loop_norm = float(detector_settings[channel]['close_loop_norm'])
        output_offset = float(detector_settings[channel]['output_offset'])
        sgamp = float(detector_settings[channel]['signal_gen_current'])
        sgfreq = float(detector_settings[channel]['signal_gen_frequency'])
        group_name = str(info[0]['group_name'])
        series_name = h5io.extract_series_name(int(info[0]['series_num']))

        # rshunt resistance
        rshunt = None
        if 'shunt_resistance' in detector_settings[channel].keys():
            rshunt = float(detector_settings[channel]['shunt_resistance'])
        elif 'rshunt' in detector_settings[channel].keys():
            rshunt = float(detector_settings[channel]['rshunt'])
        if rshunt == np.nan:
            rshunt = None
            
        # parasitic resistance 
        rp = None
        if 'parasitic_resistance' in detector_settings[channel].keys():
            rp = float(detector_settings[channel]['parasitic_resistance'])
        elif 'rp' in detector_settings[channel].keys():
            rp = float(detector_settings[channel]['rp'])
        if rp == np.nan:
            rp = None
            
                        
        # save relevant detector settings in dictionary
        data_config = dict()
        data_config['series_name'] = series_name
        data_config['group_name'] = group_name
        data_config['fs'] = fs
        data_config['output_variable_gain'] = output_gain
        data_config['output_variable_offset'] = output_offset
        data_config['close_loop_norm'] = close_loop_norm
        data_config['rshunt'] = rshunt
        data_config['rp'] = rp
        data_config['tes_bias'] = tes_bias
        data_config['sgfreq'] = sgfreq
        data_config['sgamp'] = sgamp


        # Apply cuts
        zerocut = np.all(traces!=0, axis=1)
        traces = traces[zerocut]  
    
        # cut pileup
        cut = qp.autocuts_didv(traces, fs=fs)
        traces = traces[cut]

        # Fit data
        didvobj = qp.DIDV(traces,
                          fs,
                          sgfreq, 
                          sgamp, 
                          rshunt, 
                          rp=rp,
                          dutycycle=0.5,
                          add180phase=False)
            
        # process
        didvobj.processtraces()
        
        # store
        self._didv_data[channel] = {'didvobj': didvobj,
                                    'group_name': group_name,
                                    'series_name': series_name,
                                    'base_path': base_path,
                                    'ivsweep_results': None,
                                    'data_config': data_config,
                                    'resolution': None}


        
    def _set_processed_data(self, channel,
                            didv_data,
                            overwrite=False,
                            **kwargs ):
        
        """
        Set dIdV data from already processed raw 
        data, in the form of pandas series or 
        dictionary. The function can be called multiple
        times for different channels 
        """

        # check if data can be overwritte
        if  self._didv_data is None:
             self._didv_data = dict()
        else:
            if (not overwrite and
                 channel in self._didv_data.keys()):
                raise ValueError(f'ERROR: Data for channel {channel} '
                                 'already available. Set overwrite=True')
            
        # check didv data
        if isinstance(didv_data, pd.Series):
            didv_data = didv_data.to_dict()
        elif not isinstance(didv_data, dict):
            raise ValueError(
                'ERROR: "didv_data" should be a dictionary '
                ' or pandas series')
               
        # check variables 
        available_vars = list(didv_data.keys())
        required_vars = ['avgtrace_didv', 'didvmean','didvstd',
                         'offset_didv','offset_err_didv',
                         'sgamp', 'sgfreq',
                         'fs_didv']
        
        for var in required_vars:
            if var not in available_vars:
                raise ValueError(f'ERROR: Missing variable {var} '
                                 ' in didv_data')
            
        # check if channel stored in didv_data
        if 'channel' in didv_data.keys():
            channel_didv = str(didv_data['channel'])
            if channel != channel_didv:
                raise ValueError(
                    f'ERROR: Channel name in didv data "{channel_didv}" '
                    'is different than input channel!')

        # check if other parameters are available
        rshunt = None
        if 'rshunt_noise' in didv_data.keys():
            rshunt =  didv_data['rshunt_noise']
        elif 'rshunt_didv' in didv_data.keys():
            rshunt =  didv_data['rshunt_didv']
            
        # parasitic resistance 
        rp = None
        if 'rp_noise' in didv_data.keys():
            rp =  didv_data['rp_noise']
        elif 'rp_didv' in didv_data.keys():
            rp =  didv_data['rp_didv']

            
        # duty cycle
        dutycycle = 0.5
        if 'dutycycle' in didv_data.keys():
            dutycycle = didv_data['dutycycle']

        # data config
        data_config_pars = ['series_name', 'group_name',
                            'fs', 'output_variable_gain',
                            'output_variable_offset',
                            'close_loop_norm', 'tes_bias',
                            'sgfreq','sgamp']
        
        data_config = dict()
        for config in  data_config_pars:
            config_name = config + '_didv'
            if  config_name in didv_data.keys():
                data_config[config] = didv_data[config_name]
            else:
                data_config[config]= None
                
        data_config['rshunt'] = rshunt
        data_config['rp'] = rp

                 
        # Instantiate QETpy DIDV Object
        didvobj = qp.didvinitfromdata(
            didv_data['avgtrace_didv'][:len(didv_data['didvmean'])],
            didv_data['didvmean'],
            didv_data['didvstd'],
            didv_data['offset_didv'],
            didv_data['offset_err_didv'],
            didv_data['fs_didv'],
            didv_data['sgfreq'],
            didv_data['sgamp'],
            rsh=rshunt,
            rp=rp,
            dutycycle=dutycycle,
        )


        # store
        self._didv_data[channel] = {'didvobj': didvobj,
                                    'group_name': data_config['group_name'],
                                    'series_name': data_config['series_name'],
                                    'base_path': None,
                                    'ivsweep_results': None,
                                    'data_config': data_config,
                                    'resolution': None}
        
    
