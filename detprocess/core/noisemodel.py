import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from scipy import constants
from scipy.signal import savgol_filter
from lmfit import Model
from detprocess.core.filterdata import FilterData
import qetpy as qp
import matplotlib.pyplot as plt


__all__ = ['NoiseModel']


class NoiseModel(FilterData):
    """
    Class to manage noise calculation from 
    randoms for multiple channels
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

        # data
        self._noise_data = dict()
        self._tbath = None
        self._tload_guess = None
        self._tc = dict()
        self._gta = dict()
        self._inductance = dict()
        self._poles = None

        # internal data for SC fit
        self._s_isquid_for_fit = None

        
    def set_iv_didv_results_from_file(self, file_name,
                                      poles=2,
                                      channels=None):
        """
        set dIdV and/or IV sweep results from an HDF5 file
        """

        self.load_hdf5(file_name)

        if channels is None:
            channels = self._filter_data.keys()
            if not channels:
                raise ValueError(f'ERROR: No data loaded... '
                                 f'Check file {file_name}')
        elif isinstance(channels, str):
            channels = [channels]
       
        for chan in channels:

            # check if filter data
            if chan not in self._filter_data.keys():
                raise ValueError(f'ERROR: No data loaded for channel {chan}. '
                                 f'Check file {file_name} !')
            # dIdV results
            didv_results = None
            try:
                didv_results = self.get_didv_results(chan, poles=poles)
            except:
                print(f'WARNING: No {poles}-poles dIdV results found for '
                      f'channel {chan}!')

            # IV results
            ivsweep_results = None
            try:
                ivsweep_results = self.get_ivsweep_results(chan)
            except:
                pass


            self.set_iv_didv_results_from_dict(
                chan,
                didv_results=didv_results,
                poles=poles,
                ivsweep_results=ivsweep_results
            )
   
    def set_iv_didv_results_from_dict(self, channel,
                                      didv_results=None,
                                      poles=2,
                                      ivsweep_results=None):
        """
        Set didv from dictionary for specified channel
        """

        # check if channel exist
        if channel not in self._noise_data.keys():
            self._noise_data[channel] = dict()


        # save poles
        self._poles = poles
   
        # dIdV results
        if didv_results is not None:

            if poles is None:
                raise ValueError('ERROR: dIdV poles (2 or 3) required!')

            # add to filter data
            metadata = None
            if 'metadata' in didv_results:
                metadata = didv_results['metadata']
            self.set_didv_results(
                channel,
                didv_results,
                poles=poles,
                metadata=metadata
            )

            # Add small signal parameters
            if 'smallsignalparams' in didv_results.keys():
                self._noise_data[channel]['smallsignalparams'] = (
                    didv_results['smallsignalparams'].copy()
                )
            else:
                raise ValueError(f'ERROR: dIdV fit results '
                                 f'does not contain "smallsignalparams" '
                                 f'for channel {channel}!')
            
            # add "biasparams"
            if ('biasparams' in didv_results.keys()
                and didv_results['biasparams'] is not None):
                self._noise_data[channel]['biasparams'] = didv_results['biasparams']


        # IV results
        if ivsweep_results is not None:

            # add to filter data
            self.set_ivsweep_results(
                channel,
                ivsweep_results,
                'noise')

            # add to noise data
            if 'biasparams' not in self._noise_data[channel]:
                self._noise_data[channel]['biasparams'] = ivsweep_results
            else:
                self._noise_data[channel]['biasparams'].update(ivsweep_results)

            # add more quantities
            if 'rn' not in  self._noise_data[channel]['biasparams']:
                self._noise_data[chan]['biasparams']['rn'] = (
                    ivsweep_results['rn']
                )

            # add more quantities
            if 'rp' not in  self._noise_data[channel]['biasparams']:
                self._noise_data[chan]['biasparams']['rp'] = (
                    ivsweep_results['rp']
                )
                

            if 'rshunt' not in self._noise_data[channel]['biasparams']:
                self._noise_data[channel]['biasparams']['rshunt'] = (
                    ivsweep_results['rshunt']
                )
                
            # inductance
            if 'normal_didv_fit_L' in ivsweep_results.keys():
                self.set_inductance(channel,
                                    ivsweep_results['normal_didv_fit_L'],
                                    'normal')

            if 'sc_didv_fit_L' in ivsweep_results.keys():
                self.set_inductance(channel,
                                    ivsweep_results['sc_didv_fit_L'],
                                    'sc')
                
        # check Rn
        if 'rn' not in  self._noise_data[channel]['biasparams']:
            raise ValueError(f'ERROR: No Rn found for channel {channel}! '
                             f'Add "rn" key/value in "ivsweep_results" '
                             f'argument.')
        
                
    def set_inductance(self, channel, L, state):
        """
        Set inductance when normal/SC state
        """
        if (state != 'normal' and state != 'sc'):
            raise ValueError('ERROR: "state" argument should be '
                             '"normal" or "sc"!')

        if channel not in  self._inductance.keys():
            self._inductance[channel] = dict()

        self._inductance[channel][state] = float(L)

        
    def set_tbath(self, tbath):
        """
        Set bath temperature
        """

        self._tbath = float(tbath)

    def set_tload_guess(self, tload):
        """
        Set bath temperature
        """

        self._tload_guess = float(tload)

        
    def set_tc(self, channel, tc):
        """
        Set Tc
        """

        self._tc[channel] = float(tc)

    
    def set_gta(self, channel, gta):
        """
        Set Gta for specified channel
        """

        self._gta[channel] = float(gta)
        

    def set_psd_from_file(self, file_name):
        """
        Set TES data from file
        """
        
        raise ValueError('ERROR: Not implemented yet!')

        
        # load hdf5 file
        self.load_hdf5(file_name)


    def set_psd(self, channel, psd, psd_freqs, state):
        """
        Set two-sided PSD for "normal", "sc", "transition" 
        state, The TES bias used when taking PSD data 
        needs also to  be set.
        """

        state_list = ['normal', 'sc', 'transition']
        if state not in state_list:
            raise ValueError(
                'ERROR: "state" argument should be '
                '"normal", "sc", or "transition"'
            )

        # initialized 
        if channel not in  self._noise_data.keys():
            self._noise_data[channel] = dict()
            
        if state not in self._noise_data[channel]:
            self._noise_data[channel][state] = dict()

        # check if folded
        is_folded = not np.any(psd_freqs<0)
        if is_folded:
            raise ValueError(
                'ERROR: Two-sides PSD needs '
                'to be provided, not folded PSD!')
        
        # if not is_folded:
        #     fs =  np.max(np.abs(psd_freqs))*2
         #    psd_freqs, psd = qp.foldpsd(psd, fs)
            
        self._noise_data[channel][state]['psd'] = psd
        self._noise_data[channel][state]['psd_freqs'] = psd_freqs
     

    def set_normal_fit_results(self, channel,
                               squiddc=None,
                               squidpole=None,
                               squidn=None):
        """
        Set normal noise fit results
        """

        if (squiddc is None
            or squidpole is None
            or squidn is None):
            raise ValueError('ERROR: "squiddc", "squidpole", and '
                             '"squidn" are required!')
                
        if channel not in  self._noise_data.keys():
            self._noise_data[channel] = dict()

        if 'normal' not in self._noise_data[channel]:
            self._noise_data[channel]['normal'] = dict()

        # add data
        self._noise_data[channel]['normal']['fit'] = {
            'squiddc': float(squiddc),
            'squidpole': float(squidpole),
            'squidn': float(squidn)
        }


    def set_sc_fit_results(self, channel, tload=None):
         
        """
        Set SC noise fit results
        """

        if tload is None:
            raise ValueError('ERROR: "tload" is required!')
                       
        if channel not in  self._noise_data.keys():
            self._noise_data[channel] = dict()

        if 'sc' not in self._noise_data[channel]:
            self._noise_data[channel]['sc'] = dict()
        
        # add data
        self._noise_data[channel]['sc']['fit'] = {'tload': float(tload)}


    def set_squid_noise(self, channel, squid_noise, squid_noise_freqs):
         
        """
        Set SQUID noise
        """

        if channel not in  self._noise_data.keys():
            self._noise_data[channel] = dict()

        if 'sim' not in self._noise_data[channel]:
            self._noise_data[channel]['sim'] = {'normal': {},
                                                'sc' :{},
                                                'transition': {}}

        self._noise_data[channel]['sim']['normal']['s_isquid'] = (
            squid_noise
        )
        self._noise_data[channel]['sim']['normal']['freqs'] =  (
            squid_noise_freqs
        )

        
    def calc_squid_noise(self,
                         channels=None,
                         do_fit_normal_noise=False,
                         fit_range=(100, 1e5),
                         squiddc0=6e-12, squidpole0=200, squidn0=0.7,
                         lgc_plot=False, xlims=None, ylims=None,
                         lgc_save_fig=False, save_path=None):
        """
        Calculate SQUID+Electronics noise from normal noise
        either by using normal noise directly (subtracting johnson noise)
        or fitting PSD

        Parameters
        ----------
        fit_range : tuple, optional
            The frequency range over which to do the fit.
        squiddc0 : float, optional
            Initial guess for the squiddc parameter.
        squidpole0 : float, optional
            Initial guess for the squidpole parameter.
        squidn0 : float, optional
            Initial guess for the squidn parameter.
        lgc_plot : bool, optional
            If True, a plot of the fit is shown.
        xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim().
        ylims : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim().
        lgc_save_fig : bool, optional
            If True, the figure is saved.
        save_path : str
            Directory to save data
 
        Returns
        -------
        None


        """

        # check if fit normal noise
        if do_fit_normal_noise:
            self.fit_normal_noise(
                channels=channels,
                fit_range=fit_range,
                squiddc0=squiddc0,
                squidpole0=squidpole0,
                squidn0=squidn0,
                lgc_plot=lgc_plot,
                xlims=xlims, ylims=ylims,
                lgc_save_fig=lgc_save_fig,
                save_path=save_path)
            return
        
        # check channels
        if channels is None:
            channels = self._noise_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

            
        for chan in channels:

            # check psd
            if chan not in self._noise_data.keys():
                raise ValueError(f'ERROR: No data for channel {chan} '
                                 'available. Set psd and didv data first!')
        

            # check normal parameters exists
            if ('normal' not in self._noise_data[chan]
                or 'psd' not  in self._noise_data[chan]['normal']):
                raise ValueError(f'ERROR: No normal psd for channel {chan} '
                                 'available. Set psd first!')
                
            # check tload
            if self._tload_guess is None:
                raise ValueError(f'ERROR: Normal fit requires Tload '
                                 f'Set Tload first using function '
                                 f'"set_tload_guess(tload)"!')
            

            # check tload
            if self._tbath is None:
                raise ValueError(f'ERROR: Bath temperature is needed! '
                                 f'Set Tbath first using function '
                                 f'"set_tbath(tbath)"!')
    
            # didv
            if 'biasparams' not in self._noise_data[chan]:
                raise ValueError(f'ERROR: No iv/didv data for channel {chan} '
                                 'available. Set didv data first!')
            # inductance
            if (chan not in self._inductance
                or 'normal' not in  self._inductance[chan]):
                raise ValueError(f'ERROR: No inductance value for channel {chan} '
                                 f'available. Set inductance first using function '
                                 f'set_inductance("{chan}", L, "normal")')
        
        # Loop channels and fit data
        for chan in channels:
            
            # psd array
            psd = self._noise_data[chan]['normal']['psd']
            psd_freqs =  self._noise_data[chan]['normal']['psd_freqs']
            
            # tc
            tc = self._tc[chan]
            
            # rn/rload
            rn = float(self._noise_data[chan]['biasparams']['rn'])
            rp = float(self._noise_data[chan]['biasparams']['rp'])
            rshunt = None
            if 'rsh' in self._noise_data[chan]['biasparams']:
                rshunt = self._noise_data[chan]['biasparams']['rsh']
            else:
                rshunt = self._noise_data[chan]['biasparams']['rshunt']
            rload = rp + rshunt

            # inductance
            L = float(self._inductance[chan]['normal'])
            
            # calc squid noise
            squid_noise = qp.sim.get_squid_noise_from_normal_noise(
                freqs=psd_freqs,
                normal_noise=psd,
                tload=self._tload_guess,
                tc=tc, rload=rload, rn=rn,
                inductance=L)

            if 'sim' not in self._noise_data[chan]:
                self._noise_data[chan]['sim'] = {'normal': {},
                                                 'sc' :{},
                                                 'transition': {}}

            self._noise_data[chan]['sim']['normal']['s_isquid'] = squid_noise
            self._noise_data[chan]['sim']['normal']['freqs'] =  psd_freqs
            
            
    def fit_normal_noise(self, channels=None,
                         fit_range=(100, 1e5),
                         squiddc0=6e-12, squidpole0=200, squidn0=0.7,
                         lgc_plot=False, xlims=None, ylims=None,
                         lgc_save_fig=False, save_path=None):
        """
        Function to fit the noise components of the SQUID+Electronics.
        Fits all normal noise PSDs and stores the average value for
        squiddc, squidpole, and squidn as attributes of the class.

        Parameters
        ----------
        fit_range : tuple, optional
            The frequency range over which to do the fit.
        squiddc0 : float, optional
            Initial guess for the squiddc parameter.
        squidpole0 : float, optional
            Initial guess for the squidpole parameter.
        squidn0 : float, optional
            Initial guess for the squidn parameter.
        lgc_plot : bool, optional
            If True, a plot of the fit is shown.
        xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim().
        ylims : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim().
        lgc_save_fig : bool, optional
            If True, the figure is saved.
        save_path : str
            Directory to save data
        
        Returns
        -------
        None

        """
        
        # check channels
        if channels is None:
            channels = self._noise_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

            
        for chan in channels:

            # check psd
            if chan not in self._noise_data.keys():
                raise ValueError(f'ERROR: No data for channel {chan} '
                                 'available. Set psd and didv data first!')
        

            # check normal parameters exists
            if ('normal' not in self._noise_data[chan]
                or 'psd' not  in self._noise_data[chan]['normal']):
                raise ValueError(f'ERROR: No normal psd for channel {chan} '
                                 'available. Set psd first!')
                

            # check tc
            if chan not in self._tc.keys():
                raise ValueError(f'ERROR: No Tc for channel {chan} '
                                 f'available. Set Tc first using function '
                                 f'set_tc("{chan}", tc)"!')
            

            # check tload
            if self._tload_guess is None:
                raise ValueError(f'ERROR: Normal fit requires Tload '
                                 f'Set Tload first using function '
                                 f'"set_tload_guess(tload)"!')
            

            # check tload
            if self._tbath is None:
                raise ValueError(f'ERROR: Bath temperature is needed! '
                                 f'Set Tbath first using function '
                                 f'"set_tbath(tbath)"!')
    
            # didv
            if 'biasparams' not in self._noise_data[chan]:
                raise ValueError(f'ERROR: No iv/didv data for channel {chan} '
                                 'available. Set didv data first!')
            # inductance
            if (chan not in self._inductance
                or 'normal' not in  self._inductance[chan]):
                raise ValueError(f'ERROR: No inductance value for channel {chan} '
                                 f'available. Set inductance first using function '
                                 f'set_inductance("{chan}", L, "normal")')
            
        # Loop channels and fit data
        for chan in channels:
            
            # psd array
            psd = self._noise_data[chan]['normal']['psd']
            psd_freqs =  self._noise_data[chan]['normal']['psd_freqs']
            
            # didv
            didv_results = self.get_didv_results(chan, self._poles)
            
            # tc
            tc = self._tc[chan]

            # rn/rload
            rn = float(self._noise_data[chan]['biasparams']['rn'])
            rp = float(self._noise_data[chan]['biasparams']['rp'])
            rshunt = None
            if 'rsh' in self._noise_data[chan]['biasparams']:
                rshunt = self._noise_data[chan]['biasparams']['rsh']
            else:
                rshunt = self._noise_data[chan]['biasparams']['rshunt']
            rload = rp + rshunt

            # inductance
            L = float(self._inductance[chan]['normal'])
        
            # fit range
            ind_lower = (np.abs(psd_freqs - fit_range[0])).argmin()
            ind_upper = (np.abs(psd_freqs - fit_range[1])).argmin()
         
            xdata = psd_freqs[ind_lower:ind_upper]
            ydata = self._flatten_psd(psd_freqs, psd)[ind_lower:ind_upper]

            # build model and fit
            model = Model(_normal_noise, independent_vars=['freqs'])
         
            params = model.make_params(
                squiddc=squiddc0,
                squidpole=squidpole0,
                squidn=squidn0,
                rload=rload,
                tload=self._tload_guess,
                rn=rn,
                tc=tc,
                inductance=L,
            )
            params['tc'].vary = False
            params['tload'].vary = False
            params['rload'].vary = False
            params['rn'].vary = False
            params['inductance'].vary = False
            result = model.fit(ydata, params, freqs=xdata)

            # fit result
            fitvals = result.values
           
            # store values
            self._noise_data[chan]['normal']['fit'] = fitvals

            # calculate sQUID noise
            squiddc = fitvals['squiddc'],
            squidpole = fitvals['squidpole'],
            squidn = itvals['squidn'],
            squid_noise = self.get_squid_noise_from_fit(
                psd_freqs, squiddc, squidpole, squidn
            )

            
            if 'sim' not in self._noise_data[chan]:
                self._noise_data[chan]['sim'] = {'normal': {},
                                                 'sc' :{},
                                                 'transition': {}}
            # FIXME: Need to unfold
            self._noise_data[chan]['sim']['normal']['s_isquid'] = squid_noise
            self._noise_data[chan]['sim']['normal']['freqs'] =  psd_freqs
                        
            # Instantiate Noise sim
            """
            noise_sim = qp.sim.TESnoise(
                rload=float(rload),
                r0=rn,
                rshunt=rshunt,
                inductance=L,
                beta=0,
                loopgain=0,
                tau0=0,
                G=0,
                qetbias=tes_bias,
                tc=tc,
                tload=self._tload_guess,
                tbath=self._tbath,
                squiddc=fitvals['squiddc'],
                squidpole=fitvals['squidpole'],
                squidn=fitvals['squidn'],
            )

         
            if lgc_plot:
                qp.plotting.plot_noise_sim(
                    f=psd_freqs,
                    psd=psd,
                    noise_sim=noise_sim,
                    istype='normal',
                    qetbias=round(tes_bias*1e9)/1e9,
                    lgcsave=lgc_save_fig,
                    figsavepath=save_path,
                    xlims=xlims,
                    ylims=ylims,
                )
            """

            
    def get_squid_noise_from_fit(self, freqs, squiddc, squidpole, squidn):
        """
        Calculate SQUID noise
        """

        s_isquid = (squiddc * (1.0 + (squidpole / freqs)**squidn))**2.0
        return  s_isquid

            
    def fit_sc_noise(self, channels=None,
                     fit_range=(100, 1e5),
                     lgc_plot=False, xlims=None, ylims=None,
                     lgc_save_fig=False, save_path=None):
        """
        Function to fit the components of the SC Noise. Fits all SC
        noise PSDs and stores tload 

        Parameters
        ----------
        fit_range : tuple, optional
            The frequency range over which to do the fit.

        Parameters
        ----------
        fit_range : tuple, optional
            The frequency range over which to do the fit.
        lgc_plot : bool, optional
            If True, a plot of the fit is shown.
        xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim().
        ylims : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim().
        lgc_save_fig : bool, optional
            If True, the figure is saved.
        save_path : str
            Directory to save data
        
        Returns
        -------
        None

        """
               
        # check channels
        if channels is None:
            channels = self._noise_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

            
        for chan in channels:
            if chan not in self._noise_data.keys():
                raise ValueError(f'ERROR: No data for channel {chan} '
                                 'available. Set data first!')
                
            
            # check normal parameters exists
            if ('sc' not in self._noise_data[chan]
                or 'psd' not  in self._noise_data[chan]['sc']):
                raise ValueError(f'ERROR: No SC  psd for channel {chan} '
                                 'available. Set psd first!')

            
            if ('sim' not in self._noise_data[chan]
                or 'normal' not  in self._noise_data[chan]['sim']):
                raise ValueError(f'ERROR: SQUID noise should be first calculated '
                                 f'from normal noise for channel {chan}')

                     
            if chan not in self._tc.keys():
                raise ValueError(f'ERROR: No Tc for channel {chan} '
                                 f'available. Set Tc first using function '
                                 f'"set_tc({chan}, tc)"!')

            # didv
            if 'biasparams' not in self._noise_data[chan]:
                raise ValueError(f'ERROR: No iv/didv data for channel {chan} '
                                 'available. Set didv data first!')
        
            # inductance
            if (chan not in self._inductance
                or 'sc' not in  self._inductance[chan]):
                raise ValueError(f'ERROR: No inductance value for channel {chan} '
                                 f'available. Set inductance first using function '
                                 f'set_inductance("{chan}", L, "sc")')

        if (lgc_plot
            and (self._poles != 2 or self._poles != 3)):
            print('WARNING: Unable to display SC noise model. Fit results (2 or 3-poles) '
                  'required!')
            

        # Loop channels and fit data
        for chan in channels:
            
            # psd array
            psd = self._noise_data[chan]['sc']['psd']
            psd_freqs =  self._noise_data[chan]['sc']['psd_freqs']
            is_folded = not np.any(psd_freqs<0)
            if is_folded:
                raise ValueError('ERROR: SC PSD should be two-sided')

            # fold
            fs =  np.max(np.abs(psd_freqs))*2
            psd_fold_freqs, psd_fold = qp.foldpsd(psd, fs)

            # Tc
            tc = self._tc[chan]
        
            # rn/rload
            rn = float(self._noise_data[chan]['biasparams']['rn'])
            rp = float(self._noise_data[chan]['biasparams']['rp'])
            rshunt = None
            if 'rsh' in self._noise_data[chan]['biasparams']:
                rshunt = self._noise_data[chan]['biasparams']['rsh']
            else:
                rshunt = self._noise_data[chan]['biasparams']['rshunt']
            rload = rp+rshunt
            
            # inductance
            L = self._inductance[chan]['sc']
                      
            # SQUID noise 
            squid_noise = self._noise_data[chan]['sim']['normal']['s_isquid']
            squid_noise_freqs = self._noise_data[chan]['sim']['normal']['freqs']
          
            # fold
            fs_squid =  np.max(np.abs(squid_noise_freqs))*2
            squid_noise_fold_freqs, squid_noise_fold = (
                qp.foldpsd(squid_noise, fs_squid)
            )
                    
            # fit range
            ind_lower = (np.abs(psd_freqs - fit_range[0])).argmin()
            ind_upper = (np.abs(psd_freqs - fit_range[1])).argmin()
         
            xdata = psd_fold_freqs[ind_lower:ind_upper]
            ydata = self._flatten_psd(psd_fold_freqs, psd_fold)[ind_lower:ind_upper]
            squid_noise_fold_reduced =  squid_noise_fold[ind_lower:ind_upper]

            # save internal data to pass to fit function
            # (no extra array argument allowed with lmfit)
            self._s_isquid_for_fit = squid_noise_fold_reduced
            
            # build model and fit
            model = Model(self._sc_noise, independent_vars=['freqs'])
            params = model.make_params(
                tload=self._tload_guess,
                rload=rload,
                inductance=L
            )
            
            params['rload'].vary = False
            params['inductance'].vary = False
            result = model.fit(ydata, params, freqs=xdata)

            # fit result
            fitvals = result.values

            # store values
            self._noise_data[chan]['sc']['fit'] = fitvals

            if self._verbose:
                tload = format(fitvals['tload']*1e3, '.2f')
                print(f'INFO: Fitted Tload from SC noise for '
                      f'channel {chan} = {tload} mK')


            # if 2 or 3-poles fit not availble:
            # unable to instantiate QETpy TESnoise
            if (self._poles != 2 and self._poles != 3):
                continue

            # get didv results
            didv_results = self.get_didv_results(chan, poles=self._poles)
            
                
            # Instantiate Noise sim
            noise_sim = qp.sim.TESnoise(
                freqs=psd_freqs,
                didv_result=didv_results,
                tc=tc,
                tload=fitvals['tload'],
                tbath=self._tbath,
                p0_manual=None,
                n=5.0,
                lgc_ballistic=True,
                squid_noise_current=squid_noise,
                squid_noise_current_freqs=squid_noise_freqs,
                lgc_diagnostics=True)

            # save
            s_isquid = noise_sim.s_isquid(psd_freqs)
            s_iloadsc = noise_sim.s_iloadsc(psd_freqs)
            s_itotsc = noise_sim.s_itotsc(psd_freqs)
          
            self._noise_data[chan]['sim']['sc']['s_itotsc'] = s_itotsc
            self._noise_data[chan]['sim']['sc']['s_isquid'] = s_itotsc
            self._noise_data[chan]['sim']['sc']['freqs'] = psd_freqs
            

            # fold 
            _,s_isquid_fold  = qp.foldpsd(s_isquid, fs)
            _,s_iloadsc_fold  = qp.foldpsd(s_iloadsc, fs)
            _,s_itotsc_fold = qp.foldpsd(s_itotsc, fs)
            
            # plot
            if lgc_plot:
                
                fig, ax = plt.subplots(figsize=(11, 6))
                f = psd_fold_freqs
                
                ax.grid(which="minor", linestyle="dotted", alpha=0.5)
                ax.loglog(
                    f[1:], np.sqrt(psd_fold[1:]), alpha=0.5,
                    color='#8c564b', label='Raw Data',
                )
                ax.loglog(
                    f[1:], np.sqrt(s_isquid_fold[1:]),
                    color='#9467bd', label='Squid+Electronics Noise',
                )
                ax.loglog(
                    f[1:], np.sqrt(s_iloadsc_fold[1:]),
                    color='#ff7f0e', label='Load Noise',
                )
                ax.loglog(
                    f[1:], np.sqrt(s_itotsc_fold[1:]),
                    color='#d62728', label='Total Noise',
                )
                ax.legend()
                ax.set_xlabel('Frequency [Hz]')
                ax.set_ylabel('Input Referenced Current Noise '
                              '[A/$\sqrt{\mathrm{Hz}}$]')
                ax.set_title(f'{chan} Superconducting State noise')
                ax.tick_params(which="both", direction="in",
                               right=True, top=True)

                plt.show()
            

    def analyze_noise(self, channels=None,
                      fit_range=(100, 1e5),
                      do_fit_normal_noise=False,
                      squiddc0=6e-12, squidpole0=200, squidn0=0.7,
                      lgc_plot=False,
                      xlims=None, ylims_current=None, ylims_power=None, 
                      lgc_save_fig=False, save_path=None):
        """
        Function to analyze and plot noise PSD with all the theoretical noise
        components (calculated from the didv fits). 

        lgc_plot : bool, optional
            If True, a plot of the fit is shown.
        xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim().
        ylims_current : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim() for the current noise
            plots.
        ylims_power : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim() for the power noise
            plots.
        lgc_save : bool, optional
            If True, the figure is saved.
        save_path : str
            Directory to save data

        Returns
        -------
        None 
        """

        # check channels
        if channels is None:
            channels = self._noise_data.keys()
        elif isinstance(channels, str):
            channels = [channels]

            
        for chan in channels:
            
            if chan not in self._noise_data.keys():
                raise ValueError(f'ERROR: No data for channel {chan} '
                                 'available. Set data first!')
            
            # check if either sc psd or fit available
            state = 'sc'
            if (state not in self._noise_data[chan]
                or ('psd' not in self._noise_data[chan][state]
                    and 'fit' not  in self._noise_data[chan][state])):
                raise ValueError(f'ERROR: No {state} psd or fit results for '
                                 f'channel {chan} available!')

            # check if either normal psd or fit or squid noise available
            state = 'normal'
            if ('sim' not in self._noise_data[chan]
                and state not in self._noise_data[chan]):
                raise ValueError(f'ERROR: No {state} psd or fit results '
                                 f'or squid noise available for '
                                 f'channel {chan}!')
            
            # check transition psd
            if ('transition' not in self._noise_data[chan]
                or 'psd' not in self._noise_data[chan]['transition']):
                raise ValueError(f'ERROR: No transition psd for '
                                 f'channel {chan} available!'
                                 f'Set PSD first!')
                                    
            # check tc
            if chan not in self._tc.keys():
                raise ValueError(f'ERROR: No Tc for channel {chan} '
                                 f'available. Set Tc first using function '
                                 f'"set_tc({chan}, tc)"!')
        

            # check Gta
            #if chan not in self._gta.keys() and self._verbose:
            #    print(f'WARNING: No Gta available for channel {chan}. '
            #          f'By default Gta is calculated using 5*p0/tc.\n'
            #          f'You can set more precise Gta using function '
            #          f'"set_gta({chan}, gta)"!')
                
            # didv
            #if 'biasparams' not in self._noise_data[chan]:
            #    raise ValueError(f'ERROR: No iv/didv data for channel {chan} '
            #                     'available. Set didv data first!')
            
                    
        # loop channels and do SC/normal fit if needed
        for chan in channels:

            #  get parameters
            tc = self._tc[chan]
            rn = self._noise_data[chan]['biasparams']['rn']
            rp = self._noise_data[chan]['biasparams']['rp']
            rshunt = None
            if 'rsh' in self._noise_data[chan]['biasparams']:
                rshunt = self._noise_data[chan]['biasparams']['rsh']
            else:
                rshunt = self._noise_data[chan]['biasparams']['rshunt']
            rload = rp + rshunt
            L = self._noise_data[chan]['smallsignalparams']['L']
            p0 = self._noise_data[chan]['biasparams']['p0']
            
            # SQUID  noise from normal noise
            if ('sim' not in self._noise_data[chan]
                or not self._noise_data[chan]['sim']['normal']):
                self.calc_squid_noise(
                    channels=chan,
                    do_fit_normal_noise=do_fit_normal_noise,
                    fit_range=fit_range,
                    squiddc0=squiddc0, squidpole0=squidpole0,
                    squidn0=squidn0,
                    lgc_plot=lgc_plot, xlims=xlims, ylims=ylims,
                    lgc_save_fig=lgc_save_fig, save_path=save_path
                )
                                  
            # two sided SQUID noise
            squid_noise = self._noise_data[chan]['sim']['normal']['s_isquid']
            squid_noise_freqs = self._noise_data[chan]['sim']['normal']['freqs']
            
            # Tload from SC fit
            if 'fit' not in self._noise_data[chan]['sc']:
                self.fit_sc_noise(channels=channels,
                                  fit_range=fit_range,
                                  lgc_plot=lgc_plot,
                                  xlims=xlims, ylims=ylims_current,
                                  lgc_save_fig=lgc_save_fig,
                                  save_path=save_path)

            # fit results
            fit_sc = self._noise_data[chan]['sc']['fit']

            # didv results
            didv_results = self.get_didv_results(chan, poles=self._poles)
                     
            # get psd in transition
            psd = self._noise_data[chan]['transition']['psd']
            psd_freqs =  self._noise_data[chan]['transition']['psd_freqs']
                      
            # Instantiate TES Noise
            noise_sim = qp.sim.TESnoise(
                freqs=psd_freqs,
                didv_result=didv_results,
                tc=tc,
                tload=fit_sc['tload'],
                tbath=self._tbath,
                p0_manual=None,
                n=5.0,
                lgc_ballistic=True,
                squid_noise_current=squid_noise,
                squid_noise_current_freqs=squid_noise_freqs,
                lgc_diagnostics=False)

            s_ites = noise_sim.s_ites(psd_freqs)
            s_iload = noise_sim.s_iload(psd_freqs)
            s_itfn = noise_sim.s_itfn(psd_freqs)
            s_isquid = noise_sim.s_isquid(psd_freqs)
            s_itot = noise_sim.s_itot(psd_freqs)

            s_ptes = noise_sim.s_ptes(psd_freqs)
            s_pload = noise_sim.s_pload(psd_freqs)
            s_ptfn = noise_sim.s_ptfn(psd_freqs)
            s_psquid = noise_sim.s_psquid(psd_freqs)
            s_ptot = noise_sim.s_ptot(psd_freqs)
            
            # save
            self._noise_data[chan]['sim']['transition'] = {
                's_ites':s_ites, 's_iload':s_iload,
                's_itfn':s_itfn, 's_isquid':s_isquid,
                's_itot':s_itot,
                's_ptes':s_ptes, 's_pload':s_pload,
                's_ptfn':s_ptfn, 's_psquid':s_psquid,
                's_ptot':s_ptot,
                'freqs': psd_freqs
            }
         
         
            
            # plot
            if lgc_plot:
                
                # fold
                fs =  np.max(np.abs(psd_freqs))*2
                psd_fold_freqs, psd_fold = qp.foldpsd(psd, fs)
                p_psd = psd*np.abs(noise_sim.dPdI)**2
                _,p_psd_fold  = qp.foldpsd(p_psd, fs)
                
                _,s_ites_fold  = qp.foldpsd(s_ites, fs)
                _,s_iload_fold  = qp.foldpsd(s_iload, fs)
                _,s_itfn_fold = qp.foldpsd(s_itfn, fs)
                _,s_isquid_fold = qp.foldpsd(s_isquid, fs)
                _,s_itot_fold = qp.foldpsd(s_itot, fs)

                _,s_ptes_fold  = qp.foldpsd(s_ptes, fs)
                _,s_pload_fold  = qp.foldpsd(s_pload, fs)
                _,s_ptfn_fold = qp.foldpsd(s_ptfn, fs)
                _,s_psquid_fold = qp.foldpsd(s_psquid, fs)
                _,s_ptot_fold = qp.foldpsd(s_ptot, fs)
                
                f = psd_fold_freqs
                    
                fig, ax = plt.subplots(figsize=(11, 6))
                ax.grid(which="major", linestyle='--')
                ax.grid(which="minor", linestyle="dotted", alpha=0.5)
                ax.tick_params(which="both", direction="in", right=True, top=True)
                ax.set_xlabel(r'Frequency [Hz]')
                ax.set_title(f'{chan} Current Noise')

                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_ites_fold[1:])),
                    color='#1f77b4',
                    linewidth=1.5,
                    label='TES Johnson Noise',
                )
                
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_iload_fold[1:])),
                    color='#ff7f0e',
                    linewidth=1.5,
                    label='Load Noise',
                )
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_itfn_fold[1:])),
                    color='#2ca02c',
                    linewidth=1.5,
                    label='TFN Noise',
                )
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_itot_fold[1:])),
                    color='#d62728',
                    linewidth=1.5,
                    label='Total Noise',
    )
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_isquid_fold[1:])),
                    color='#9467bd',
                    linewidth=1.5,
                    label='Squid+Electronics Noise',
                )
                
                ax.loglog(f[1:], np.sqrt(psd_fold[1:]),
                          color='#8c564b', alpha=0.8,
                          label='Raw Data')
                
                ax.set_ylabel('TES Current Noise $[A/\sqrt{\mathrm{Hz}}]$')

                lgd = plt.legend(loc='upper right')


                # power nouise
                fig, ax = plt.subplots(figsize=(11, 6))
                ax.grid(which="major", linestyle='--')
                ax.grid(which="minor", linestyle="dotted", alpha=0.5)
                ax.tick_params(which="both", direction="in", right=True, top=True)
                ax.set_xlabel(r'Frequency [Hz]')
                ax.set_title(f'{chan} Power Noise')

                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_ptes_fold[1:])),
                    color='#1f77b4',
                    linewidth=1.5,
                    label='TES Johnson Noise',
                )
                
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_pload_fold[1:])),
                    color='#ff7f0e',
                    linewidth=1.5,
                    label='Load Noise',
                )
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_ptfn_fold[1:])),
                    color='#2ca02c',
                    linewidth=1.5,
                    label='TFN Noise',
                )
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_ptot_fold[1:])),
                    color='#d62728',
                    linewidth=1.5,
                    label='Total Noise',
    )
                ax.loglog(
                    f[1:],
                    np.sqrt(np.abs(s_psquid_fold[1:])),
                    color='#9467bd',
                    linewidth=1.5,
                    label='Squid+Electronics Noise',
                )
                
                ax.loglog(f[1:], np.sqrt(p_psd_fold[1:]), color='#8c564b',
                          alpha=0.8, label='Raw Data')

                ax.set_ylabel(r'Input Referenced Power Noise [W/$\sqrt{\mathrm{Hz}}$]')
             
                lgd = plt.legend(loc='upper right')

                plt.show()
                
    def _flatten_psd(self, f, psd):
        """
        Helper function to smooth out all the spikes in a single-sided psd
        in order to more easily fit the SC and Normal state noise.
        
        Parameters
        ----------
        f: ndarray
          Array of frequency values.
        psd : ndarray
          Array of one sided psd values.
    
        Returns
        -------
        flattened_psd : ndarray
         Array of values of smoothed psd.
        
        """
        
        sav = np.zeros(psd.shape)
        div = int(.0025*len(psd))
        sav_lower = savgol_filter(psd[1:], 3, 1, mode='interp', deriv=0)
        sav_upper = savgol_filter(psd[1:], 45, 1, mode='interp', deriv=0)
        sav[1:div+1] = sav_lower[:div]
        sav[1+div:] = sav_upper[div:]
        sav[0] = psd[0]
        flattened_psd = qp.utils.make_decreasing(sav, x=f)
        
        return flattened_psd
    
    def _sc_noise(self, freqs, tload, rload, inductance):
        """
        Functional form of the Super Conducting state noise. Including
        the Johnson noise for the load resistor and the SQUID + downstream
        electronics noise. See qetpy.sim.TESnoise class for more info.
        
        Parameters
        ----------
        freqs : array
         Array of frequencies.
        tload : float
          The temeperature of the load resistor in Kelvin.
        rload : float
          Value of the load resistor in Ohms.
        inductance : float
          The inductance of the TES line.
        s_isquid : array
          SQUID+Electronics noise array 

        Returns
        -------
        s_tot : array
           Array of values corresponding to the theoretical SC state noise.
        
        """

        omega = 2.0 * np.pi * freqs
        dIdVsc = 1.0 / (rload + 1.0j * omega * inductance)
        s_vload = 4.0 * constants.k * tload * rload * np.ones_like(freqs)    
        s_iloadsc = s_vload * np.abs(dIdVsc)**2.0
        return s_iloadsc + self._s_isquid_for_fit


    def _normal_noise(self, freqs, squiddc, squidpole,
                      squidn, rload, tload, rn, tc,
                      inductance):
        """
        Functional form of the normal state noise. Including the Johnson
        noise for the load resistor, the Johnson noise for the TES, and the
        SQUID + downstream electronics noise. See qetpy.sim.TESnoise class
        for more info.
        
        Parameters
        ----------
        freqs : array
          Array of frequencies.
        squiddc : float
           The average value for the white noise from the squid (ignoring
           the 1/f component).
        squidpole : float
          The knee for the 1/f component of the noise.
        squidn : float
            The factor for the 1/f^n noise.
        rload : float
           Value of the load resistor in Ohms.
        tload : float
           The temeperature of the load resistor in Kelvin.
        rn : float
          The value of the resistance of the TES when normal.
        tc : float
           The SC transistion temperature of the TES.
        inductance : float
           The inductance of the TES line.
        
        Returns
        -------
        s_tot : array
           Array of values corresponding to the theoretical normal state noise.
        
        """

        omega = 2.0 * np.pi * freqs
        dIdVnormal = 1.0 / (rload + rn + 1.0j * omega * inductance)
        
        # Johnson noise for the load resistor
        s_vload = 4.0 * constants.k * tload * rload * np.ones_like(freqs)
        s_iloadnormal = s_vload * np.abs(dIdVnormal)**2.0
        
        # Johnson noise for the TES
        s_vtesnormal = 4.0 * constants.k * tc * rn * np.ones_like(freqs)
        s_itesnormal = s_vtesnormal * np.abs(dIdVnormal)**2.0
        
        # SQUID + downstream electronics noise
        s_isquid = (squiddc * (1.0 + (squidpole / freqs)**squidn))**2.0
        
        # total noise
        s_tot = s_iloadnormal + s_itesnormal + s_isquid
        
        return s_tot
