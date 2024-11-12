import numpy as np
import os
import vaex as vx
import pandas as pd
import qetpy as qp
from math import log10, floor
from glob import glob
from pathlib import Path
import types
import cloudpickle
import pytesdaq.io as h5io
import math
import array
from detprocess.core.oftrigger import OptimumFilterTrigger
from detprocess.process.randoms import Randoms
from detprocess.core.filterdata import FilterData
from qetpy.utils import convert_channel_name_to_list,convert_channel_list_to_name
from scipy.signal import correlate
from scipy import stats, signal, interpolate, special, integrate

__all__ = [
    'Salting'
]


class Salting(FilterData):
    """
    Class for injecting salt into datasets for multiple channels. Can be used to
    understand cut efficiencies.

    Attributes
    ----------
    asd:asdfasdfasdfasdfasdfasdf

    """

    def __init__(self, filter_file, didv_file=None, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """

        # initialize raw data dictionary
        self._series = None
        self._group_name = None
        self._raw_base_path = None
        self._series_list = None
        self._detector_config = None
        self._ivdidv_data = dict()
        self._saltarraydict = dict()

        
        # intialize randoms dataframe
        self._dataframe = None
        self._listofdfs = []

        # intialize event list
        self._event_list = None

        # sample rate stored for convenience
        self._fs = None
        
        # store the energies from the DM spectra that you have sampled
        self._DMenergies = np.array([])
        #self._Channelenergies = np.array([])
        
        self._verbose = verbose

        super().__init__(verbose=verbose)

        self._filter_file = filter_file
        self._didv_file = didv_file
        
        self.load_hdf5(filter_file, overwrite=False)
        if didv_file is not None:
            self.load_hdf5(didv_file, overwrite=False)


        
    def get_detector_config(self, channel):
        """
        get detector config
        """
        if self._detector_config is None:
            print('WARNING: No data has been set yet! '
                  'Returning None ')
            return None
        elif channel not in self._detector_config.keys():
            print(f'WARNING: No channel {channel}  found! '
                  f'Returning None ')
            return None
        return self._detector_config[channel]
                
    def get_sample_rate(self):
        """
        Get sample rate in Hz ("calc_psd" needs to be 
        called before)
        """

        return self._fs
        
            
    def _generate_randoms(self, raw_path, series=None,
                         random_rate=None,
                         nevents=None,
                         min_separation_msec=20,
                         edge_exclusion_msec=50,
                         restricted=False,
                         calib=False,
                         ncores=1):
        """
        Generate randoms from continuous data
        """
        
        self._dataframe = None

        # generate randoms
        rand_inst = Randoms(raw_path, series=series,
                            verbose=False,
                            restricted=restricted,
                            calib=calib)

        
        self._dataframe = rand_inst.process(
            random_rate=random_rate,
            nrandoms=nevents,
            min_separation_msec=min_separation_msec,
            edge_exclusion_msec=edge_exclusion_msec,
            lgc_save=False,
            lgc_output=True,
            ncores=ncores
        )
            
   
    def set_raw_data_path(self,group_path,series,restricted):
        
        self._series = series
        self._raw_base_path = group_path
    
    
    def sample_DMpdf(self,function, xrange, nsamples=1000, npoints=10000, normalize_cdf=True):
        """
        Produces randomly sampled values based on the arbitrary PDF defined
        by `function`, done using inverse transform sampling.

        Parameters
        ----------
        function : FunctionType
            The 1D probability density function to be randomly sampled from.
        xrange : array_like
            A 1D array of length 2 that defines the range over which the PDF
            in `function` is defined. Outside of this range, it is assumed that
            the PDF is zero.
        nsamples : int, optional
            The number of random samples that we wish to create from the PDF
            defined by `function`.
        npoints : int, optional
            The number of points to use in the numerical integration to evaluate
            the CDF of `function`. This is also the number of points used in the
            interpolation of the inverse of the CDF.
        normalize_cdf : bool, optional
            Boolean value to normalize the CDF or not. If True, the CDF is normalized
            by the PDF area. If False, no normalization is done.

        Returns
        -------
        rvs : ndarray
            The random samples that were taken from the inputted PDF defined by
            `function`. This is a 1D array of length `nsamples`.

        Raises
        ------
        TypeError
            If inputted `function` is not of FunctionType

        Notes
        -----
        For a discussion of inverse transform sampling, see the Wikipedia page:
            https://en.wikipedia.org/wiki/Inverse_transform_sampling

        """
        if not isinstance(function, types.FunctionType):
            raise TypeError("Inputted variable function is not FunctionType.")

        x = np.linspace(xrange[0], xrange[1], num=npoints)
        pdf = function(x)

        cdf = integrate.cumtrapz(pdf, x=x, initial=0.0)

        if normalize_cdf:
            cdf /= cdf[-1]

        inv_cdf = interpolate.interp1d(cdf, x)

        samples = np.random.rand(nsamples)
        sampled_energies = inv_cdf(samples)

        self._DMenergies = np.append(self._DMenergies,sampled_energies* 1e3) #this is hardcoded! This is because the dRdE spectrum I'm using is in keV!
        return sampled_energies

    def get_DMenergies(self):
        return self._DMenergies
    
    def clear_DMenergies(self):
        self._DMenergies = np.array([])
        
    def channel_energy_split(self,mean=0.5, std_dev=0.2, npairs=10):
        #make n pairs which will be the same as the number of events to sim
        listofsplits = []
        for i in range(npairs):
            # Generate random numbers from a Gaussian distribution
            random_numbers = np.random.normal(loc=mean, scale=std_dev, size=2)
            
            # Clip values to be between 0 and 1
            random_numbers = np.clip(random_numbers, 0, 1)
            
            # Check if the sum is positive (important for the normalization step)
            if np.sum(random_numbers) > 0:
                # Normalize to sum to 1
                random_numbers = random_numbers / np.sum(random_numbers)
            
            listofsplits.extend([random_numbers])
            #self._Channelenergies = listofsplits
        return listofsplits

    def get_energy_perchannel(self):
        return self._Channelenergies
    
    def set_energy_splits_to1(self,energysplits):
        for sublist in energysplits:
            for i in range(len(sublist)):
                sublist[i] = 1
        return energysplits

    def generate_salt(self,channels,noise_tag, template_tag , dpdi_tag,dpdi_poles,energies,pdf_file,PCE,nevents = 100):
        channel_list  = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nb_channels = len(channel_list)
       
        # get template 1D or 2D array
        template, time_array = self.get_template(channel_name, tag=template_tag)
        nb_samples = template.shape[-1]
        # get psd/csd:
        if len(channel_list)>1: csd, csd_freqs = self.get_csd(channel_name, tag=noise_tag)
        else: csd, csd_freqs = self.get_psd(channel_name, tag=noise_tag)
        #get filt template
        tempinst = OptimumFilterTrigger(trigger_channel=channel_name, fs=1.25e6, template=template, noisecsd=csd, pretrigger_samples=12500)
        if len(channel_list) > 1 :templates_td = template.squeeze(axis=1)
        else: templates_td = template
        tempinst.update_trace(templates_td)
        filttemplate = tempinst.get_filtered_trace()
        #setup the output dict  
        salt_var_dict = {'salt_template_tag': list(),
                         'salt_recoil_energy_eV': list(),
                         'saltchanname': list()}
        base_keys = ['salt_amplitude', 'salt_filt_amplitude',  'salt_energy_eV']
        # get dpdi for each individual channels
        
        dpdi_dict = {}
        for chan in channel_list:
            dpdi, _= self.get_dpdi(chan, poles = dpdi_poles, tag=dpdi_tag)
            dpdi_dict[chan] = dpdi
        if pdf_file and energies:
            raise ValueError('Only pass either list of energies or DM PDFs, not both!')
        #get the energies 
        if pdf_file:
            masses = []
            salt_var_dict['salt_dm_mass_MeV'] = []
            self.clear_DMenergies()
            with open(pdf_file, 'rb') as f:
                dmdists = cloudpickle.load(f)
            for mass, data in dmdists.items():
                dmrate_function = data["dmrate"]
                masses.append(mass)
                self.sample_DMpdf(dmrate_function,[1e-5,1],nsamples = nevents)
                salt_var_dict['salt_dm_mass_MeV'].extend([mass] * nevents)
            DM_energies = self.get_DMenergies()
            nevents = len(DM_energies)
        if energies:
            if not isinstance(energies, list):
                energies = [energies]
            DM_energies = [energy for energy in energies for _ in range(nevents)]
            nevents = len(DM_energies)
               
        # generate the random selections in time 
        series = self._series
        cont_data = self._raw_base_path
        sep_time = nb_samples/1.25e6
        self._generate_randoms(cont_data, series=None, nevents=nevents,
                               min_separation_msec=sep_time, ncores=4)

        # Create channel-specific keys
        for key in base_keys:
            for chan in channel_list:
                salt_var_dict[f'{key}_{chan}'] = [[] for _ in range(nevents)]
        #get the scaling factors for the template

        #this includes fraction of deposited energy in each channel and PCE
        if nb_channels > 1:
            #salts = np.zeros((nb_events,2))
            energiesplits = self.channel_energy_split(npairs=nevents)    
            unitysplits = True
            if unitysplits is True:
                 energiesplits = self.set_energy_splits_to1(energiesplits)
            #get the template to use for the salt
            salts = [[] for _ in range(nevents)]
            filtsalts = [[] for _ in range(nevents)]
            for i,chan in enumerate(channel_list):
                dpdi = dpdi_dict[chan]
                temp = template[i]
                norm_energy = qp.get_energy_normalization(time_array, temp[0], dpdi=dpdi[0], lgc_ev=True)
                scaled_template = temp[0]/norm_energy
                for n in range(nevents):
                    if 'single' in template_tag: 
                        fullyscaled_template = scaled_template * DM_energies[n]
                        scaledfilttemplate = filttemplate * DM_energies[n]
                    else: 
                        fullyscaled_template = scaled_template * DM_energies[n]*energiesplits[n][i]*PCE[i]
                        scaledfilttemplate = filttemplate[0] * DM_energies[n]*energiesplits[n][i]*PCE[i]
                        
                    salts[n].append([fullyscaled_template])   
                    filtsalts[n].append([scaledfilttemplate]) 
                    if 'saltarray' not in self._saltarraydict:
                        self._saltarraydict['saltarray'] = []
                        self._saltarraydict['filtsaltarray'] = []
                        self._saltarraydict['timearray'] = []  
                    if len(self._saltarraydict['saltarray']) <= n:
                        self._saltarraydict['saltarray'].append([])
                        self._saltarraydict['filtsaltarray'].append([])
                        self._saltarraydict['timearray'].append([])
                    if len(salt_var_dict['salt_template_tag']) <= n:
                        salt_var_dict['salt_template_tag'].append([])
                        salt_var_dict['salt_recoil_energy_eV'].append([])
                        salt_var_dict['saltchanname'].append([])
                        
                    self._saltarraydict['saltarray'][n].append(fullyscaled_template)
                    self._saltarraydict['filtsaltarray'][n].append(scaledfilttemplate)
                    self._saltarraydict['timearray'][n].append(time_array)

                    salt_var_dict[f'salt_amplitude_{chan}'][n] = max(fullyscaled_template)
                    salt_var_dict[f'salt_energy_eV_{chan}'][n] = DM_energies[n]*energiesplits[n][i]
                    salt_var_dict[f'salt_filt_amplitude_{chan}'][n] = max(scaledfilttemplate) 
                    salt_var_dict[f'salt_template_tag'][n] = template_tag
                    salt_var_dict[f'salt_recoil_energy_eV'][n] = DM_energies[n]
                    salt_var_dict[f'saltchanname'][n] = channel_name
        else: 
            salts = []
            filtsalts = []
            dpdi = dpdi_dict[chan]
            norm_energy = qp.get_energy_normalization(time_array, template, dpdi = dpdi[0], lgc_ev=True)
            scaled_template = template/norm_energy
            for n in range(nevents):
                if 'single' in template_tag: 
                    fullyscaled_template = scaled_template * DM_energies[n]
                    filttemplate = filttemplate[0] * DM_energies[n]
                else: 
                    fullyscaled_template = scaled_template * DM_energies[n]*PCE
                    scaledfilttemplate = filttemplate[0] * DM_energies[n]*PCE
                salts.append(fullyscaled_template)
                filtsalts.append(scaledfilttemplate)
                if 'saltarray' not in self._saltarraydict:
                    self._saltarraydict['saltarray'] = [] 
                    self._saltarraydict['timearray'] = []
                    self._saltarraydict['filtsaltarray'] = []
                if len(self._saltarraydict['saltarray']) <= n:
                    self._saltarraydict['saltarray'].append([])
                    self._saltarraydict['filtsaltarray'].append([])
                    self._saltarraydict['timearray'].append([])
                if len(salt_var_dict['salt_template_tag']) <= n:
                    salt_var_dict['salt_template_tag'].append([])
                    salt_var_dict['salt_recoil_energy_eV'].append([])     
                    salt_var_dict['saltchanname'].append([])          

                self._saltarraydict['saltarray'].append(fullyscaled_template)
                self._saltarraydict['filtsaltarray'].append(scaledfilttemplate)
                self._saltarraydict['timearray'].append(time_array)

                salt_var_dict[f'salt_amplitude_{chan}'][n] = max(fullyscaled_template)
                salt_var_dict[f'salt_energy_eV_{chan}'][n] = DM_energies[n]
                salt_var_dict[f'salt_filt_amplitude_{chan}'][n] = max(scaledfilttemplate)
                salt_var_dict[f'salt_template_tag'][n] = template_tag
                salt_var_dict[f'salt_recoil_energy_eV'][n] = DM_energies[n]
                salt_var_dict[f'saltchanname'][n] = channel_name
                
        maxlen = len(self._dataframe) 
        for key in salt_var_dict:
            salt_var_dict[key] = salt_var_dict[key][:maxlen]   
        df = vx.from_dict(salt_var_dict)
        self._dataframe = self._dataframe.join(df)
        self._listofdfs.append(self._dataframe)
        if len(self._listofdfs) > 1:
            pandas_dfs = []
            
            for df in self._listofdfs:
                # Flatten any multi-dimensional columns to ensure compatibility with pandas
                for col in df.get_column_names():
                    if df[col].ndim > 1:
                        # Convert multi-dimensional columns to a string representation or summary
                        df[col] = df[col].apply(lambda x: str(x))
                
                # Convert the vaex DataFrame to pandas after flattening
                pandas_dfs.append(df.to_pandas_df())
            
            # Concatenate using pandas to handle missing columns and fill NaNs with 0
            combined_pandas_df = pd.concat(pandas_dfs, axis=0, join='outer').fillna(0)
            
            # Convert the result back to a vaex DataFrame
            self._dataframe = vx.from_pandas(combined_pandas_df)        
        
        return salts,filtsalts  
    
    def set_dataframe(self, dataframe=None):
        """
        Set raw data path and vaex dataframe 
        with randoms events (either dataframe directly
        or path to vaex hdf5 files)
        """
        
        # initialize data
        #self.clear_dataframe()
        # check dataframe
        # check filter data
        if self._dataframe:
            print('WARNING: Some salt have been previously generated.')
        if dataframe is not None:
            
            if isinstance(dataframe, vx.dataframe.DataFrame):
                if len(dataframe)<1:
                    raise ValueError('ERROR: No event found in the datafame!')
            else:
                dataframe = self._load_dataframe(dataframe)

            self._dataframe = dataframe

    def get_dataframe(self):
        return self._dataframe
    
    def inject_raw_salt(self, channels, trace, seriesID, eventID):
        """
        FIXME
        """

        # initialize salted traces
        newtraces = []
        
        # channels 
        channel_list  = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nb_channels = len(channel_list)

        # traces
        is_list_input = isinstance(trace, list)
        trace_array = np.array(trace, copy=False) if is_list_input else trace
        if trace_array.ndim == 1:
            trace_array = trace_array.reshape(1, trace_array.shape[-1])

        # check dimensions
        if nb_channels != trace_array.shape[0]:
            raise ValueError('ERROR:  number of channels incompatible with array shape!')
                    
        # filter dataframe
        filtered_df = None
        if ((self._dataframe['event_number'] == eventID).count() > 0
            and (self._dataframe['series_number'] == seriesID).count() > 0):
            filtered_df = (
                self._dataframe[(self._dataframe['event_number'] == eventID)
                                & (self._dataframe['series_number'] == seriesID)]
            )
        else:
            return []


        # loop channels
        for i, waveform in enumerate(trace_array):

            # channel name
            chan = channel_list[i]
            
            # initialize
            newtrace = np.array(waveform, copy=True)
            salts_before_ADC = np.zeros(np.shape(waveform),dtype=float)

            # loop row of filter datafame and add salt
            columns_to_extract = ['salt_template_tag', f'salt_amplitude_{chan}',
                                  'trigger_index','saltchanname']
            for _, j in filtered_df[columns_to_extract].to_pandas_df().iterrows():
                template_tag = j['salt_template_tag']
                tempchan = j['saltchanname']
                template,times = self.get_template(tempchan, tag=template_tag)
                if "|" in tempchan:
                    temp = template[i][0]
                else:
                    temp = template
                nb_samples=len(times)
                saltamp = j[f'salt_amplitude_{chan}']
                saltpulse = temp* saltamp
                simtime = j['trigger_index']
                salt_and_baseline = saltpulse+newtrace[0]
                salt_and_baseline -= salt_and_baseline[0]
                salts_before_ADC[simtime:simtime+nb_samples] += salt_and_baseline
                newtrace += salts_before_ADC
            newtraces.append(newtrace)
        if is_list_input:
            return [list(trace) for trace in newtraces]
        else:
            return np.array(newtraces)
        #return newtraces



