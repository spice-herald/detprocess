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

__all__ = ['NoiseModel']


def _normal_noise(freqs, squiddc, squidpole,
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


def _sc_noise(freqs, tload, squiddc, squidpole, squidn, rload, inductance):
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
    squiddc : float
        The average value for the white noise from the squid (ignoring
        the 1/f component).
    squidpole : float
        The knee for the 1/f component of the noise.
    squidn : float
        The factor for the 1/f^n noise.
    rload : float
        Value of the load resistor in Ohms.
    inductance : float
        The inductance of the TES line.

    Returns
    -------
    s_tot : array
        Array of values corresponding to the theoretical SC state noise.

    """

    omega = 2.0 * np.pi * freqs
    dIdVsc = 1.0 / (rload + 1.0j * omega * inductance)
    s_vload = 4.0 * constants.k * tload * rload * np.ones_like(freqs)    
    s_iloadsc = s_vload * np.abs(dIdVsc)**2.0 
    s_isquid = (squiddc * (1.0 + (squidpole / freqs)**squidn))**2.0
    return s_iloadsc + s_isquid


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

        
    def set_didv_data_from_file(self, file_name, channels=None):
        """
        set didv
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
            
            # check if channel exist
            if chan not in self._noise_data.keys():
                self._noise_data[chan] = dict()
                
            # get didv results for poles=2
            data = self.get_didv_results(chan, poles=2)
            if 'smallsignalparams' in data.keys():
                self._noise_data[chan]['smallsignalparams'] = (
                    data['smallsignalparams']
                )
            else:
                raise ValueError(f'ERROR: dIdV 2-poles fit results '
                                 f'does not contain "smallsignalparams" '
                                 f'for channel {chan}')

            if ('biasparams' in data.keys()
                and data['biasparams'] is not None):
                self._noise_data[chan]['biasparams'] = (
                    data['biasparams']
                )
            else:
                raise ValueError(f'ERROR: dIdV 2-poles fit results '
                                 f'does not contain "biasparams" '
                                 f'for channel {chan}')


            # case 'rn' not in "biasparams", get IV sweep results
            if 'rn' not in  data['biasparams']:
                ivdata = None
                try:
                    ivdata = self.get_ivsweep_results(chan)
                except:
                    raise ValueError(f'ERROR: No Rn found for channel '
                                     f'{chan}. IV sweep results need to be '
                                     f'added in hdf5 file!')
                else:
                    self._noise_data[chan]['biasparams']['rn'] = ivdata['rn']
            

    def set_didv_data_from_dict(self, channel, didv_data, ivsweep_result=None):
        """
        Set didv from dictionary for specified channel
        """


        # check if channel exist
        if channel not in self._noise_data.keys():
            self._noise_data[channel] = dict()

        if 'smallsignalparams' in didv_data.keys():
            self._noise_data[channel]['smallsignalparams'] = (
                didv_data['smallsignalparams']
            )
        else:
            raise ValueError(f'ERROR: dIdV fit results '
                             f'does not contain "smallsignalparams" '
                             f'for channel {channel}')

        if ('biasparams' in didv_data.keys()
            and didv_data['biasparams'] is not None):
            self._noise_data[channel]['biasparams'] = didv_data['biasparams']
        elif ivsweep_result is not None:
            self._noise_data[channel]['biasparams'] = ivsweep_result
            # back compatibility
            self._noise_data[channel]['biasparams']['rsh'] = ivsweep_result['rshunt']
        else:
            raise ValueError(f'ERROR: dIdV fit results '
                             f'does not contain "biasparams" '
                             f'for channel {channel}!')

        # case 'rn' not in "biasparams", get IV sweep results
        if 'rn' not in self._noise_data[channel]['biasparams']:  
            if (ivsweep_result is not None and 'rn' in ivsweep_result):
                self._noise_data[channel]['biasparams']['rn'] = (
                    ivsweep_result['rn']
                )
            else:
                raise ValueError(f'ERROR: No Rn found for channel '
                                 f'{channel}. Add "rn" in "ivsweep_result" argument!')
            

    def set_tbath(self, tbath):
        """
        Set bath temperature
        """

        self._tbath = tbath

    def set_tload_guess(self, tload):
        """
        Set bath temperature
        """

        self._tload_guess = tload

        
    def set_tc(self, channel, tc):
        """
        Set Tc
        """

        self._tc[channel] = tc

    
    def set_gta(self, channel, gta):
        """
        Set Gta for specified channel
        """

        self._gta[channel] = gta
        

    def set_psd_from_file(self, file_name):
        """
        Set TES data from file
        """
        
        
        # load hdf5 file
        self.load_hdf5(file_name)


    def set_psd(self, channel, psd, psd_freqs, tes_bias,
                state):
        """
        Set PSD for "normal", "sc", "transition" state,
        The TES bias used when taking PSD data needs also to 
        be set.
        """

        state_list = ['normal', 'sc', 'transition']
        if state not in state_list:
            raise ValueError(
                'ERROR: "state" argument should be '
                '"normal", "sc", or "transition"'
            )

        
        if channel not in  self._noise_data.keys():
            self._noise_data[channel] = {'normal': None,
                                         'sc': None,
                                         'transition': None}

        # add data
        if self._noise_data[channel][state] is None:
            self._noise_data[channel][state] = dict()


        # fold psd if needed
        is_folded = not np.any(psd_freqs<0)
        
        if not is_folded:
            fs =  np.max(np.abs(psd_freqs))*2
            psd_freqs, psd = qp.foldpsd(psd, fs)
            
        self._noise_data[channel][state]['psd_fold'] = psd
        self._noise_data[channel][state]['psd_fold_freqs'] = psd_freqs
        self._noise_data[channel][state]['tes_bias'] = tes_bias


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
            self._noise_data[channel] = {'normal': None,
                                         'sc': None,
                                         'transition': None}

        if self._noise_data[channel]['normal'] is None:
            self._noise_data[channel]['normal'] = dict()

        # add data
        self._noise_data[channel]['normal']['fit'] = dict()
        self._noise_data[channel]['normal']['fit']['squiddc'] = squiddc
        self._noise_data[channel]['normal']['fit']['squidpole'] = squidpole
        self._noise_data[channel]['normal']['fit']['squidn'] = squidn


    def set_sc_fit_results(self, channel, tload=None):
         
        """
        Set SC noise fit results
        """

        if tload is None:
            raise ValueError('ERROR: "tload" is required!')
                
        if channel not in  self._noise_data.keys():
            self._noise_data[channel] = {'normal': None,
                                         'sc': None,
                                         'transition': None}

        if self._noise_data[channel]['sc'] is None:
            self._noise_data[channel]['sc'] = dict()

        # add data
        self._noise_data[channel]['sc']['fit'] = dict()
        self._noise_data[channel]['sc']['fit']['tload'] = tload

        
        
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
            if self._noise_data[chan]['normal'] is None:
                raise ValueError(f'ERROR: No normal psd for channel {chan} '
                                 'available. Set psd first!')
                

            # check tc
            if chan not in self._tc.keys():
                raise ValueError(f'ERROR: No Tc for channel {chan} '
                                 f'available. Set Tc first using function '
                                 f'"set_tc({chan}, tc)"!')
            

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
    
            
        # Loop channels and fit data
        for chan in channels:

            # psd array
            psd = self._noise_data[chan]['normal']['psd_fold']
            psd_freqs =  self._noise_data[chan]['normal']['psd_fold_freqs']
            tes_bias = self._noise_data[chan]['normal']['tes_bias']
                      
            # tc
            tc = self._tc[chan]

            # rn/rload
            rn = self._noise_data[chan]['biasparams']['rn']
            rp = self._noise_data[chan]['biasparams']['rp']
            rshunt = None
            if 'rsh' in self._noise_data[chan]['biasparams']:
                rshunt = self._noise_data[chan]['biasparams']['rsh']
            else:
                rshunt = self._noise_data[chan]['biasparams']['rshunt']
            rload = rp+rshunt

            # inductance
            L = self._noise_data[chan]['smallsignalparams']['L']
        
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
       
            # Instantiate Noise sim
            noise_sim = qp.sim.TESnoise(
                rload=rload,
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

    def fit_sc_noise(self, channels=None,
                     fit_range=(100, 1e5),
                     lgc_plot=False, xlims=None, ylims=None,
                     lgc_save_fig=False, save_path=None):
        """
        Function to fit the components of the SC Noise. Fits all SC
        noise PSDs and stores the average value for tload as an
        attribute of the class.

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
                

            if self._noise_data[chan]['sc'] is None:
                raise ValueError(f'ERROR: No sc psd for channel {chan} '
                                 'available. Set psd first!')

            if (self._noise_data[chan]['normal'] is None
                  or 'fit' not in  self._noise_data[chan]['normal']):
                raise ValueError(f'ERROR: Fit of normal noise should be done first '
                                 f'for channel {chan}')
            

            if chan not in self._tc.keys():
                raise ValueError(f'ERROR: No Tc for channel {chan} '
                                 f'available. Set Tc first using function '
                                 f'"set_tc({chan}, tc)"!')

            # didv
            if 'biasparams' not in self._noise_data[chan]:
                raise ValueError(f'ERROR: No iv/didv data for channel {chan} '
                                 'available. Set didv data first!')
        
            
        # Loop channels and fit data
        for chan in channels:

            # normal fit
            fit_normal = self._noise_data[chan]['normal']['fit']
            
            # psd array
            psd = self._noise_data[chan]['sc']['psd_fold']
            psd_freqs =  self._noise_data[chan]['sc']['psd_fold_freqs']
            tes_bias = self._noise_data[chan]['sc']['tes_bias']

            # Tc
            tc = self._tc[chan]
        
            # rn/rload
            rn = self._noise_data[chan]['biasparams']['rn']
            rp = self._noise_data[chan]['biasparams']['rp']
            rshunt = None
            if 'rsh' in self._noise_data[chan]['biasparams']:
                rshunt = self._noise_data[chan]['biasparams']['rsh']
            else:
                rshunt = self._noise_data[chan]['biasparams']['rshunt']
            rload = rp+rshunt
            
            # inductance
            L = self._noise_data[chan]['smallsignalparams']['L']
            
            # fit range
            ind_lower = (np.abs(psd_freqs - fit_range[0])).argmin()
            ind_upper = (np.abs(psd_freqs - fit_range[1])).argmin()

            # fit data
            xdata = psd_freqs[ind_lower:ind_upper]
            ydata = self._flatten_psd(psd_freqs, psd)[ind_lower:ind_upper]
            
            # build model and fit
            model = Model(_sc_noise, independent_vars=['freqs'])
            params = model.make_params(
                tload=self._tload_guess,
                squiddc=fit_normal['squiddc'],
                squidpole=fit_normal['squidpole'],
                squidn=fit_normal['squidn'],
                rload=rload,
                inductance=L,
            )
            
            params['squiddc'].vary = False
            params['squidpole'].vary = False
            params['squidn'].vary = False
            params['rload'].vary = False
            params['inductance'].vary = False
            result = model.fit(ydata, params, freqs=xdata)

            # fit result
            fitvals = result.values

            # store values
            self._noise_data[chan]['sc']['fit'] = fitvals
       
            # Instantiate Noise sim
            noise_sim = qp.sim.TESnoise(
                rload=rload,
                r0=0.0001,
                rshunt=rshunt,
                inductance=L,
                beta=0,
                loopgain=0,
                tau0=0,
                G=0,
                qetbias=tes_bias,
                tc=tc,
                tload=fitvals['tload'],
                tbath=self._tbath,
                squiddc=fit_normal['squiddc'],
                squidpole=fit_normal['squidpole'],
                squidn=fit_normal['squidn'],
            )

         
            if lgc_plot:
                qp.plotting.plot_noise_sim(
                    f=psd_freqs,
                    psd=psd,
                    noise_sim=noise_sim,
                    istype='sc',
                    qetbias=round(tes_bias*1e9)/1e9,
                    lgcsave=lgc_save_fig,
                    figsavepath=save_path,
                    xlims=xlims,
                    ylims=ylims,
                )


    def analyze_noise(self, channels=None,
                      fit_range=(100, 1e5),
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
                
            else:
                states = ['transition', 'normal', 'sc']
                for state in states:
                    if self._noise_data[chan][state] is None:
                        raise ValueError(f'ERROR: No {state} psd for '
                                         f'channel {chan} available. '
                                         f'Set psd first!')
                                    
            # check tc
            if chan not in self._tc.keys():
                raise ValueError(f'ERROR: No Tc for channel {chan} '
                                 f'available. Set Tc first using function '
                                 f'"set_tc({chan}, tc)"!')
        

            # check Gta
            if chan not in self._gta.keys() and self._verbose:
                print(f'WARNING: No Gta available for channel {chan}. '
                      f'By default Gta is calculated using 5*p0/tc.\n'
                      f'You can set more precise Gta using function '
                      f'"set_gta({chan}, gta)"!')
                
            # didv
            if 'biasparams' not in self._noise_data[chan]:
                raise ValueError(f'ERROR: No iv/didv data for channel {chan} '
                                 'available. Set didv data first!')
        
                    
        # loop channels and do SC/normal fit if needed
        for chan in channels:

            # do normal fit if not done
            if 'fit' not in self._noise_data[chan]['normal']:
                self.fit_normal_noise(channels=channels,
                                      fit_range=fit_range,
                                      squiddc0=squiddc0,
                                      squidpole0=squidpole0,
                                      squidn0=squidn0,
                                      lgc_plot=lgc_plot,
                                      xlims=xlims, ylims=ylims_current,
                                      lgc_save_fig=lgc_save_fig,
                                      save_path=save_path)

            # do SC fit if not done
            if 'fit' not in self._noise_data[chan]['sc']:
                self.fit_sc_noise(channels=channels,
                                  fit_range=fit_range,
                                  lgc_plot=lgc_plot,
                                  xlims=xlims, ylims=ylims_current,
                                  lgc_save_fig=lgc_save_fig,
                                  save_path=save_path)

            # fit results
            fit_sc = self._noise_data[chan]['sc']['fit']
            fit_normal = self._noise_data[chan]['normal']['fit']
               
            # get psd in transition
            psd = self._noise_data[chan]['transition']['psd_fold']
            psd_freqs =  self._noise_data[chan]['transition']['psd_fold_freqs']
            tes_bias = self._noise_data[chan]['transition']['tes_bias']

            # tc and gta
            tc = self._tc[chan]

            # rn/rload/r0
            r0 = self._noise_data[chan]['biasparams']['r0']
            rn = self._noise_data[chan]['biasparams']['rn']
            rp = self._noise_data[chan]['biasparams']['rp']
            p0 = self._noise_data[chan]['biasparams']['p0']
            rshunt = None
            if 'rsh' in self._noise_data[chan]['biasparams']:
                rshunt = self._noise_data[chan]['biasparams']['rsh']
            else:
                rshunt = self._noise_data[chan]['biasparams']['rshunt']
            rload = rp + rshunt

            # gta
            gta = 5*p0/tc 
            if chan in  self._gta:
                gta = self._gta[chan]
       
            
            # smallsignalparams
            L = self._noise_data[chan]['smallsignalparams']['L']
            loopgain = self._noise_data[chan]['smallsignalparams']['l']
            beta = self._noise_data[chan]['smallsignalparams']['beta']
            tau0 = self._noise_data[chan]['smallsignalparams']['tau0']
            
            # Instantiate TES Noise
            noise_sim = qp.sim.TESnoise(
                rload=rload,
                r0=r0,
                rshunt=rshunt,
                inductance=L,
                beta=beta,
                loopgain=loopgain,
                tau0=tau0,
                G=gta,
                qetbias=tes_bias,
                tc=tc,
                tload=fit_sc['tload'],
                tbath=self._tbath,
                squiddc=fit_normal['squiddc'],
                squidpole=fit_normal['squidpole'],
                squidn=fit_normal['squidn'],
            )


            # plot
            if lgc_plot:
                 qp.plotting.plot_noise_sim(
                    f=psd_freqs,
                    psd=psd,
                    noise_sim=noise_sim,
                    istype='current',
                    qetbias=round(tes_bias*1e9)/1e9,
                    lgcsave=lgc_save_fig,
                    figsavepath=save_path,
                    xlims=xlims,
                    ylims=ylims_current,
                 )

                 qp.plotting.plot_noise_sim(
                     f=psd_freqs,
                     psd=psd,
                     noise_sim=noise_sim,
                     istype='power',
                     qetbias=round(tes_bias*1e9)/1e9,
                     lgcsave=lgc_save_fig,
                     figsavepath=save_path,
                     xlims=xlims,
                     ylims=ylims_power,
                 )

                                
               
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
