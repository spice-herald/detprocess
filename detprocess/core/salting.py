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
import math
import array
from detprocess.core.oftrigger import OptimumFilterTrigger
from detprocess.process.randoms import Randoms, RawData
from detprocess.core.filterdata import FilterData
from qetpy.utils import convert_channel_name_to_list,convert_channel_list_to_name
from scipy import integrate, interpolate
from pprint import pprint
import pytesio as h5io
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

    def __init__(self, filter_file, didv_file=None, template_info=None, verbose=True):
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
        
        # store the energies from the spectra that you have sampled
        self._energies = np.array([])
        #self._Channelenergies = np.array([])
        
        self._verbose = verbose

        super().__init__(verbose=verbose)

        self._filter_file = filter_file
        self._didv_file = didv_file
        self._template_info = template_info
        
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

            
    
    def sample_pdf(self,function, xrange, nsamples=1000, npoints=10000, normalize_cdf=True):
        """
        Produces randomly sampled values based on the arbitrary PDF defined
        by `function`, done using inverse transform sampling.

        Parameters
        ----------
        function : FunctionType
            The 1D probability density function to be randomly sampled from.
        xrange : array_like
            A 1D array of length 2 that defines the range (in ev) over which the PDF
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
        
        #this is hardcoded! This is because the dRdE spectrum I'm using is in eV!
        self._energies = np.append(self._energies,sampled_energies)
        
        return sampled_energies

    def get_sampled_energies(self):
        return self._energies
    
    def clear_sampled_energies(self):
        self._energies = np.array([])
        
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

    def generate_salt(self, channels, template_tag, dpdi_tag, dpdi_poles,
                      PCE, energies = None, pdf_file = None, pdf_tag = None, pdf_bounds = [1e-2, 1e3],
                      nevents = None, rate = None, poisson = False,
                      do_salt_deadtime=False,
                      livetime=None):
        """
        Generate salting metadata
        
        Parameters
        ----------

        channels : list
            list of channels to receive the salt

        template_tag : string
            tag for pulse template in the filter file 
        
        dpdi_tag : string
            tag for dpdi in the filter file   

        dpdi_poles : string
            tag for dpdi pole # in the filter file 

        energies : float or list, optional
            list of energies at which to salt   
        
        pdf_file : string, optional
            path to file containing DM/LEE PDFs
        
        pdf_tag : string, optional
            tag corresponding to the specfic PDF to pull. If "DM", 
            this will interpret the file as containing a variety of masses
            and run an individual salting on each one.

    

        nevents : integer, optional
            # of salted pulses to inject

        rate : float, optional
            Rate (in Hz) of salted events 

        poisson : bool, optional
            add poisson fluctuations on the number of salts to inject (defalt no)

        """

        if nevents is None and rate is None and pdf_tag != 'DM':
            raise ValueError('User must specify either a number of samples, or a sample rate (unless salting with DM spectra)')
        elif pdf_tag == 'DM':
            pass
        elif nevents is None: #calculate the # of salts based on dataset length.
            nevents = rate * self._rawdata_inst.get_duration()

        if nevents is not None and poisson:
            nevents = np.random.poisson(nevents)

        
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

        if livetime is not None:
            salt_var_dict['salting_livetime'] = list()

        
        base_keys = ['salt_amplitude', 'salt_energy_eV']
        # get dpdi for each individual channels
        
        dpdi_dict = {}
        if dpdi_tag and dpdi_poles:
            for chan in channel_list:
                dpdi, _= self.get_dpdi(chan, poles = dpdi_poles, tag=dpdi_tag)
                dpdi_dict[chan] = dpdi


        if pdf_file is not None and energies is not None:
            raise ValueError('You can either pass DM PDFs, LEE PDFs, or discrete energies. Pick one!')

        #get the energies 
        if pdf_file:
            if pdf_tag == 'DM':
                if nevents is not None or rate is not None:
                    print('Warning: ignoring nevents/rate argument; dark matter rate is pre-defined and stored in the PDF files')
                masses = []
                salt_var_dict['salt_dm_mass_MeV'] = []
                self.clear_sampled_energies()
                with open(pdf_file, 'rb') as f:
                    dmdists = cloudpickle.load(f)
                for mass, data in dmdists.items():
                    dmrate_function = data["dmrate"]
                    masses.append(mass)
                    self.sample_pdf(dmrate_function,pdf_bounds,nsamples = nevents)
                    salt_var_dict['salt_dm_mass_MeV'].extend([mass] * nevents)
                sampled_energies = self.get_sampled_energies()
                nevents = len(sampled_energies)
            else:
                self.clear_sampled_energies()
                salt_var_dict[f'salt_{pdf_tag}'] = []
                with open(pdf_file, 'rb') as f:
                    PDF_function = cloudpickle.load(f)[pdf_tag]
                self.sample_pdf(PDF_function,pdf_bounds, nsamples = nevents, npoints = int(2e5))
                salt_var_dict[f'salt_{pdf_tag}'].extend([pdf_tag] * nevents)
                sampled_energies = self.get_sampled_energies()

        if energies:
            if not isinstance(energies, list):
                energies = [energies]
            sampled_energies = [energy for energy in energies for _ in range(nevents)]
            nevents = len(sampled_energies)
               
        # generate the random selections in time 

        #if we're salting for understaning dE'/dE, disallow pileup
        if pdf_tag is None:
            sep_time = 1000*nb_samples/self._fs
        #if we're salting for non-detector physics, allow piluep
        else:
            sep_time = 0
        if self._dataframe is None:
            if do_salt_deadtime:
                self._generate_randoms(nevents=nevents,
                                       min_separation_msec=sep_time,
                                       edge_exclusion_msec=0)
            else:
                edge_exclusion = self._template_info['max_edge_exclusion']
                self._generate_randoms(nevents=nevents,
                                       min_separation_msec=sep_time,
                                       edge_exclusion_msec=edge_exclusion)
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
                temp = template[i]
                if dpdi_dict:
                    dpdi = dpdi_dict[chan]
                    norm_energy = qp.get_energy_normalization(time_array, temp[0], dpdi=dpdi[0], lgc_ev=True)
                    scaled_template = temp[0]/norm_energy
                else: scaled_template = temp[0]/max(temp[0])
                for n in range(nevents):
                    fullyscaled_template = scaled_template * sampled_energies[n]*PCE[i]

                    salts[n].append([fullyscaled_template])   
                    if len(salt_var_dict['salt_template_tag']) <= n:
                        salt_var_dict['salt_template_tag'].append([])
                        salt_var_dict['salt_recoil_energy_eV'].append([])
                        salt_var_dict['saltchanname'].append([])
                        salt_var_dict[f'salting_type'].append([])
                        if livetime is not None:
                            salt_var_dict[f'salting_livetime'].append([])
                        
                    salt_var_dict[f'salt_amplitude_{chan}'][n] = max(fullyscaled_template)
                    salt_var_dict[f'salt_energy_eV_{chan}'][n] = sampled_energies[n]
                    salt_var_dict[f'salt_template_tag'][n] = template_tag
                    salt_var_dict[f'salt_recoil_energy_eV'][n] = sampled_energies[n]
                    salt_var_dict[f'saltchanname'][n] = channel_name
                    if pdf_file:
                        if pdf_tag == 'DM':
                            salt_var_dict[f'salting_type'][n] = 'dm_pdf'
                        else:
                            salt_var_dict[f'salting_type'][n] = 'LEE_pdf'
                    else:
                        salt_var_dict[f'salting_type'][n] = f'energy_{sampled_energies[n]}_eV'
                    if livetime is not None:
                        salt_var_dict[f'salting_livetime'][n] = livetime
                        
        else: 
            salts = []
            if dpdi_dict:
                dpdi = dpdi_dict[chan]
                norm_energy = qp.get_energy_normalization(time_array, template, dpdi = dpdi[0], lgc_ev=True)
                scaled_template = template/norm_energy
            else: scaled_template = template
            for n in range(nevents):
                fullyscaled_template = scaled_template * sampled_energies[n]*PCE
                salts.append(fullyscaled_template)
                if len(salt_var_dict['salt_template_tag']) <= n:
                    salt_var_dict['salt_template_tag'].append([])
                    salt_var_dict['salt_recoil_energy_eV'].append([])     
                    salt_var_dict['saltchanname'].append([])          
                    salt_var_dict[f'salting_type'].append([]) 
                    if livetime is not None:
                        salt_var_dict[f'salting_livetime'].append([])
                        
                salt_var_dict[f'salt_amplitude_{chan}'][n] = max(fullyscaled_template)
                salt_var_dict[f'salt_energy_eV_{chan}'][n] = sampled_energies[n]
                salt_var_dict[f'salt_template_tag'][n] = template_tag
                salt_var_dict[f'salt_recoil_energy_eV'][n] = sampled_energies[n]
                salt_var_dict[f'saltchanname'][n] = channel_name
                if pdf_file:
                    if pdf_tag == 'DM':
                        salt_var_dict[f'salting_type'][n] = 'dm_pdf'
                    else:
                        salt_var_dict[f'salting_type'][n] = 'LEE_pdf'
                else:
                    salt_var_dict[f'salting_type'][n] = f'energy_{sampled_energies[n]}_eV'

                if livetime is not None:
                    salt_var_dict[f'salting_livetime'][n] = livetime
                    
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
        
        if self._dataframe:
            print('WARNING: Some salt have been previously generated and will be ovewritten')
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
                template, times, template_metdata = self.get_template(tempchan, tag=template_tag, return_metadata=True)
                pretrigger = template_metdata['nb_pretrigger_samples']
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
                simtime = int(trigger_index)
   

                segment = saltpulse[pretrigger:]            
                end = min(simtime + len(segment), len(newtrace))
                segment = segment[: end - simtime]  # trim if necessary
                newtrace[simtime:end] += segment
                    
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
    



    def _load_dataframe(self, dataframe_path):
        """
        Load vaex dataframe
        """


        # get list of files
        files_dict, base_path, group_name = (
            self._get_file_list(dataframe_path,
                                is_raw=False)
        )

        file_list = list()
        for series,files in files_dict.items():
            if len(files)>0:
                file_list.extend(files)

        dataframe = None
        if file_list:
            dataframe = vx.open_many(file_list)
        else:
            raise ValueError('ERROR: No vaex file found. Check path!')
        
        return dataframe



            
    def _get_file_list(self, file_path,
                       series=None,
                       is_raw=True,
                       restricted=False,
                       calib=False):
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

        restricted : boolean
            if True, use restricted data 
            if False (default), exclude restricted data

        Return
        -------
        
        series_dict : dict 
          list of files for splitted inot series

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
                            file_list.extend(glob(a_path + '/' + file_name_wildcard))
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

      
        # convert to series dictionary so can be easily split
        # in multiple cores
        
        series_dict = dict()
        h5reader = h5io.H5Reader()
        series_name = None
        file_counter = 0
        
        for afile in file_list:

            file_name = str(Path(afile).name)
                        
            # skip if filter file
            if 'filter_' in file_name:
                continue

            # skip didv
            if ('didv_' in file_name
                or 'iv_' in file_name):
                continue
                      
            if 'treshtrig_' in file_name:
                continue

            # calibration
            if (calib
                and 'calib_' not in file_name):
                continue

            # not calibration
            if not calib:
                
                if 'calib_' in file_name:
                    continue
                            
                # restricted
                if (restricted
                    and 'restricted' not in file_name):
                    continue

                # not restricted
                if (not restricted
                    and 'restricted' in file_name):
                    continue
                      
            # append file if series already in dictionary
            if (series_name is not None
                and series_name in afile
                and series_name in series_dict.keys()):

                if afile not in series_dict[series_name]:
                    series_dict[series_name].append(afile)
                    file_counter += 1
                continue
            
            # get metadata
            if is_raw:
                metadata = h5reader.get_metadata(afile)
                series_name = h5io.extract_series_name(metadata['series_num'])
            else:
                sep_start = file_name.find('_I')
                sep_end = file_name.find('_F')
                series_name = file_name[sep_start+1:sep_end]
                              
            if series_name not in series_dict.keys():
                series_dict[series_name] = list()

            # append
            if afile not in series_dict[series_name]:
                series_dict[series_name].append(afile)
                file_counter += 1
       
            
        if self._verbose:
            msg = ' raw data file(s) from '
            if not is_raw:
                msg = ' dataframe file(s) from '
                
            print('INFO: Found total of '
                  + str(file_counter)
                  + msg
                  + str(len(series_dict.keys()))
                  + ' different series number!')

      
        return series_dict, base_path, group_name