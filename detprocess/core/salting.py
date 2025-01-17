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
from detprocess.process.randoms import Randoms, RawData
from detprocess.core.filterdata import FilterData
from qetpy.utils import convert_channel_name_to_list,convert_channel_list_to_name
from pprint import pprint
import pyarrow as pa
import warnings
warnings.filterwarnings('ignore')
vx.settings.main.thread_count = 1
vx.settings.main.thread_count_io = 1
pa.set_cpu_count(1)

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
        self._rawdata_inst = None
        self._detector_config = None
        self._restricted = False
        self._ivdidv_data = dict()
        
        
        # intialize randoms dataframe
        self._dataframe = None
        self._injecttimes = None
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
        
            
    def _generate_randoms(self, nevents=None,
                          min_separation_msec=20,
                          edge_exclusion_msec=25,
                          ncores=1):
        """
        Generate randoms from continuous data
        """
        if self._dataframe is not None:
            del self._dataframe
        self._dataframe = None

        # generate randoms self._series = series
        rand_inst = Randoms(self._rawdata_inst, series=self._series,
                            verbose=False,
                            restricted=self._restricted,
                            calib=False)

        
        self._dataframe = rand_inst.process(
            nrandoms=nevents,
            min_separation_msec=min_separation_msec,
            edge_exclusion_msec=edge_exclusion_msec,
            lgc_save=False,
            lgc_output=True,
            ncores=ncores
        )

        print(f'INFO: {len(self._dataframe)} salting events randomly selected!')
             
        self._injecttimes = self._dataframe

            
   
    def set_raw_data(self, raw_data, series=None, restricted=False):
        """
        Set raw data path
        """

        self._series = series
        self._restricted = restricted
        self._rawdata_inst = None
        
        if isinstance(raw_data, str):
            
            self._rawdata_inst = RawData(raw_data,
                                         data_type='cont',
                                         series=series,
                                         restricted=restricted)
        else:

            if 'RawData' not in str(type(raw_data)):
                raise ValueError(
                    'ERROR: raw data argument should be either '
                    'a directory or RawData object'
                )
            
            self._rawdata_inst = raw_data

            if self._rawdata_inst.restricted != restricted:
                raise ValueError(f'ERROR: Unable to use RawData '
                                 f'object. It needs requirement restricted = '
                                 f'{self._restricted}!')
            
        # sample rate
        metadata = self._rawdata_inst.get_data_config()
        for itseries in metadata.keys():
            self._fs = metadata[itseries]['overall']['sample_rate']
            break
            
     
        # display
        if self._verbose:
            print('INFO: Data used for salting generation:')
            self._rawdata_inst.describe()

            
    
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
        
        #this is hardcoded! This is because the dRdE spectrum I'm using is in keV!
        self._DMenergies = np.append(self._DMenergies,sampled_energies* 1e3)
        
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

    def generate_salt(self, channels, noise_tag, template_tag, dpdi_tag, dpdi_poles,
                      energies, pdf_file, PCE, nevents=100):
        """
        Generate salting metadata
        """
        
        channel_list  = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nb_channels = len(channel_list)
        
        # get template 1D or 2D array
        template, time_array = self.get_template(channel_name, tag=template_tag)
        nb_samples = template.shape[-1]
        
        #setup the output dict  
        salt_var_dict = {'salt_template_tag': list(),
                         'salt_recoil_energy_eV': list(),
                         'saltchanname': list(),
                         'salting_type':list()}
        
        base_keys = ['salt_amplitude', 'salt_energy_eV']
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
        sep_time = 1000*nb_samples/self._fs
        if self._dataframe is None:
            self._generate_randoms(nevents=nevents,
                                   min_separation_msec=sep_time)
        nevents = len(self._dataframe)
        # Create channel-specific keys
        for key in base_keys:
            for chan in channel_list:
                salt_var_dict[f'{key}_{chan}'] = [[] for _ in range(nevents)]
        #get the scaling factors for the template

        #this includes fraction of deposited energy in each channel and PCE
        if nb_channels > 1:
            #get the template to use for the salt
            salts = [[] for _ in range(nevents)]
            for i,chan in enumerate(channel_list):
                dpdi = dpdi_dict[chan]
                temp = template[i]
                norm_energy = qp.get_energy_normalization(time_array, temp[0], dpdi=dpdi[0], lgc_ev=True)
                scaled_template = temp[0]/norm_energy
                for n in range(nevents):
                    fullyscaled_template = scaled_template * DM_energies[n]*PCE[i]
                    salts[n].append([fullyscaled_template])   
                    if len(salt_var_dict['salt_template_tag']) <= n:
                        salt_var_dict['salt_template_tag'].append([])
                        salt_var_dict['salt_recoil_energy_eV'].append([])
                        salt_var_dict['saltchanname'].append([])
                        salt_var_dict[f'salting_type'].append([])
                        
                    salt_var_dict[f'salt_amplitude_{chan}'][n] = max(fullyscaled_template)
                    salt_var_dict[f'salt_energy_eV_{chan}'][n] = DM_energies[n]
                    salt_var_dict[f'salt_template_tag'][n] = template_tag
                    salt_var_dict[f'salt_recoil_energy_eV'][n] = DM_energies[n]
                    salt_var_dict[f'saltchanname'][n] = channel_name
                    if pdf_file:
                        salt_var_dict[f'salting_type'][n] = 'dm_pdf'
                    else:
                        salt_var_dict[f'salting_type'][n] = f'energy_{DM_energies[n]}_eV'
        else: 
            salts = []
            dpdi = dpdi_dict[chan]
            norm_energy = qp.get_energy_normalization(time_array, template, dpdi = dpdi[0], lgc_ev=True)
            scaled_template = template/norm_energy
            for n in range(nevents):
                fullyscaled_template = scaled_template * DM_energies[n]*PCE
                salts.append(fullyscaled_template)
                if len(salt_var_dict['salt_template_tag']) <= n:
                    salt_var_dict['salt_template_tag'].append([])
                    salt_var_dict['salt_recoil_energy_eV'].append([])     
                    salt_var_dict['saltchanname'].append([])          
                    salt_var_dict[f'salting_type'].append([])
                    
                salt_var_dict[f'salt_amplitude_{chan}'][n] = max(fullyscaled_template)
                salt_var_dict[f'salt_energy_eV_{chan}'][n] = DM_energies[n]
                salt_var_dict[f'salt_template_tag'][n] = template_tag
                salt_var_dict[f'salt_recoil_energy_eV'][n] = DM_energies[n]
                salt_var_dict[f'saltchanname'][n] = channel_name
                if pdf_file:
                    salt_var_dict[f'salting_type'][n] = 'dm_pdf'
                else:
                    salt_var_dict[f'salting_type'][n] = f'energy_{DM_energies[n]}_eV'
                
        maxlen = len(self._dataframe) 
        for key in salt_var_dict:
            salt_var_dict[key] = salt_var_dict[key][:maxlen]   
        df = vx.from_dict(salt_var_dict)
        
        self._dataframe = self._dataframe.join(df)
        #if pdf_file:
        #    self._listofdfs.append(self._dataframe)
        #    self._dataframe = vx.concat(self._listofdfs)
            #self._dataframe = self.merge_dataframe(self._listofdfs)
            
        # clear dictionary
        salt_var_dict.clear()
            
        return salts  

       
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
    
    def clear_dataframe(self):
        self._dataframe = None
    
    def get_injectiontimes(self):
        return self._injecttimes

    def inject_raw_salt(self, channels, trace, seriesID, eventID,
                        include_metadata=False):
        """
        Inject salting trace into raw data
        """
        # Initialize salted traces
        newtraces = []
        
        # Convert channels to list and name
        channel_list = convert_channel_name_to_list(channels)
        nb_channels = len(channel_list)

        # Copy the trace array
        trace_array = trace.copy()
        
        # Ensure trace_array is 2D
        if trace_array.ndim == 1:
            trace_array = trace_array.reshape(1, trace_array.shape[-1])

        # Check dimensions
        if nb_channels != trace_array.shape[0]:
            raise ValueError('ERROR: number of channels incompatible with array shape!')

        # Filter the DataFrame for the given eventID and seriesID
        filtered_df = self._dataframe[
            (self._dataframe['event_number'] == eventID) &
            (self._dataframe['series_number'] == seriesID)
        ]

        
        # Check if filtered DataFrame is empty
        if filtered_df.count() == 0:
           
            # No salting needed -> return original trace
            if include_metadata:
                return trace, {}
            else:
                return trace

        # Extract common data once
        common_columns = ['salt_template_tag', 'trigger_index',
                          'saltchanname', 'salting_type']
        
        common_data = {}
        for col in common_columns:

            # Extract data as NumPy arrays
            data = filtered_df.evaluate(col, array_type='numpy')

            # Check if data is a masked array
            if np.ma.isMaskedArray(data):
                # Fill masked values with np.nan
                data = data.filled(None)

            common_data[col] = data
        

        # Extract salting type once (assuming it's the same for all entries)
        salting_types = common_data['salting_type']
        salting_type = salting_types[0] if len(salting_types) > 0 else None

        
        # Loop over each channel
        for idx_channel, waveform in enumerate(trace_array):
            
            # Get the channel name
            chan = channel_list[idx_channel]
                        
            # Initialize the new trace for this channel
            newtrace = waveform.copy()

            # Check if the amplitude column exists for this channel
            amplitude_column = f'salt_amplitude_{chan}'
            if amplitude_column not in filtered_df.get_column_names():
                print(f'WARNING: No channel {chan} found in salt df! '
                      f'Assuming single channel salt and moving on!')
                continue

            # Extract amplitude data for this channel
            amplitude_data = filtered_df.evaluate(amplitude_column, array_type='numpy')
            if np.ma.isMaskedArray(amplitude_data):
                amplitude_data = amplitude_data.filled(np.nan)

            # Iterate over the indices of the filtered DataFrame
            for idx in range(len(filtered_df)):

                # check if amplitude 
                saltamp = amplitude_data[idx]

                # Check for missing or invalid amplitude
                if np.isnan(saltamp):
                    continue
                else:
                    saltamp = float(saltamp)
                                
                # get data
                template_tag = str(common_data['salt_template_tag'][idx])
                tempchan = str(common_data['saltchanname'][idx])
                trigger_index = int(common_data['trigger_index'][idx])

                # Retrieve the template and times
                template, times = self.get_template(tempchan, tag=template_tag)
                nb_samples = len(times)

                # Handle tempchan containing '|'
                if '|' in tempchan:
                    tempchan_list = convert_channel_name_to_list(tempchan)
                    if chan in tempchan_list:
                        index = tempchan_list.index(chan)
                        temp = template[index][0]
                    else:
                        raise ValueError(f'ERROR in inject function: '
                                         f'{chan} not part of  salting channel {tempchan}. '
                                         f'Is this correct?')
                else:
                    temp = template
                
                # Add salting pulse
                saltpulse = temp * saltamp
                simtime = trigger_index
                newtrace[simtime:simtime + nb_samples] += saltpulse

            newtraces.append(newtrace)

        # Prepare output metadata
        output_metadata = {
            'salting_type': salting_type,
            'series_number': seriesID,
            'event_number': eventID
        }
        
        output_trace = np.array(newtraces)
     
        if include_metadata:
            return output_trace, output_metadata
        else:
            return output_trace
    
