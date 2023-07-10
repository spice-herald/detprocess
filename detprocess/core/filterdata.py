import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io



class FilterData:
    """
    Class to manage filter data 
    """

    def __init__(self, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """
        self._verbose = verbose
      
        # filter file data dictionary
        self._filter_data = dict()

        # available tags, parameters
        self._filter_names = dict()
        
        
        


    @property
    def verbose(self):
        return self._verbose
        
    @verbose.setter
    def verbose(self, value):
        self._verbose=value


        
        
    def describe(self):
        """
        Print filter data info
         
        Parameters:
        ----------
        None

        Return
        -------
        None
        """

        if not self._filter_data:
            print('No filter data available!')
            return

        # Let's first loop channel and get tags/display msg
        filter_display = dict()
        parameter_list = ['psd_fold', 'psd', 'template']

        for chan, chan_dict in self._filter_data.items():
            
            if chan not in  filter_display.keys():
                filter_display[chan] = dict()
                
            for par_name, val in chan_dict.items():

                # check if metadata
                if 'metadata' in par_name:
                    continue
                              
                for base_par in parameter_list:
                    
                    if base_par in par_name:
                        tag = par_name.replace(base_par, '')
                        if tag and tag[0] == '_':
                            tag  = tag[1:]
                        else:
                            tag = 'default'
                        
                        if tag not in filter_display[chan]:
                            filter_display[chan][tag] = list()
                            
                        msg = base_par
                        if isinstance(val, pd.Series):
                            msg += ': pandas.Series '
                        elif  isinstance(val, pd.DataFrame):
                            msg += ': pandas.DataFrame '
                        else: 
                            msg += str(type(val)) + ' '
                        msg += str(val.shape)

                        filter_display[chan][tag].append(msg)
                        break
                
        # loop and display
        for chan, chan_vals in filter_display.items():
            print('\nChannel ' + chan + ':')
            for tag, tag_info in chan_vals.items():
                print(' * Tag "' + tag + '":')
                if tag_info:
                    for msg in tag_info:
                        print('    ' + msg)
                                    

                
    def load_hdf5(self, file_name, overwrite=False):
        """
        Load filter data from file. Key may be overwritten if 
        already exist
 
        
        Parameters:
        ----------

        file_name : str
           filter file name 
        overwrite : boolean
           if True, overwrite exising data

        Return
        -------
        None

        """

        filter_io = h5io.FilterH5IO(file_name)
        data = filter_io.load()

        if self._verbose:
            print('INFO: Loading filter data from file '
                  + file_name)


        
        # update
        if overwrite or not self._filter_data:
            self._filter_data.update(data)
        else:
            for key, item in data.items():
                if key not in self._filter_data:
                    self._filter_data[key] = item
                    continue
                for par_name, value in item.items():
                    if par_name not in self._filter_data[key].keys():
                        self._filter_data[key][par_name] = (
                            data[key][par_name]
                        )

                        
                        
    def save_hdf5(self, file_name, overwrite=False):
        """
        Save filter data to HDF file. Key may be overwritten if 
        already exist
 
        
        Parameters:
        ----------

        file_name : str
           filter file name 
        overwrite : boolean
           if True, overwrite exising data

        Return
        -------
        None
        """

        if self._verbose:
            print('INFO: Saving noise data to file '
                  + file_name)
            if overwrite:
                print('INFO: channel data with same tag may be overwritten')

        
        filter_io = h5io.FilterH5IO(file_name)
        filter_io.save_fromdict(self._filter_data,
                                overwrite=overwrite)

        


        
    def get_psd(self, channel, tag='default', fold=False):
        """
        Get PSD for a specific channel

        Parameters:
        ----------

        channel :  str 
           channel name
        
        tag : str, optional
            psd tag, default: No tag

        fold : boolean, option
            if True, return folded psd

        Return
        ------

        psd : ndarray, 
            psd in units of amps
        f  : ndarray
            psd frequencies

        """

        
        # parameter name
        psd_name = 'psd' + '_' + tag
        if fold:
            psd_name = 'psd_fold' + '_' + tag
            
        # check if available
        if ((channel not in self._filter_data.keys()
             or (channel in self._filter_data.keys()
                 and psd_name not in self._filter_data[channel].keys()))):
            print('ERROR: No psd available for channel '
                  + channel)
            return None, None
        

        psd_series = self._filter_data[channel][psd_name]
        freq = psd_series.index
        val = psd_series.values

        return val, freq
            
          
    
    def get_template(self, channel, tag='default'):
        """
        Get template for a specific channel

        Parameters:
        ----------

        channel :  str 
           channel name
        
        tag : str, optional
            psd tag, default: No tag

        Return
        ------

        template : ndarray, 
            psd in units of amps
        time  : ndarray
            time array
        """

        
        # parameter name
        template_name = 'template' + '_' + tag
                
        # check if available
        if ((channel not in self._filter_data.keys()
             or (channel in self._filter_data.keys()
                 and template_name not in self._filter_data[channel].keys()))):
            print('ERROR: No template available for channel ' + channel
                  + '. Load from file first?')
            return None, None
        

        template_series = self._filter_data[channel][template_name]
        time = template_series.index
        val = template_series.values

        return val, time
