from detprocess.core.filterdata import FilterData
from detprocess.core.noise import Noise
from detprocess.core.template import Template
from detprocess.core.didv import DIDVAnalysis


class FilterBuilder:
    """
    Orchestrator class for building a shared filter-data object
    using Noise, Template, and DIDVAnalysis processors.

    All processors share the same internal _filter_data dictionary.
    """

    def __init__(self, verbose=True,
                 auto_save_hdf5=False,
                 didv_file_path_name=None):
        self._verbose = verbose

        # shared persistent store
        self._filter_store = {}

        # generic store API
        self.store = FilterData(verbose=verbose,
                                filter_data=self._filter_store)

        # processors sharing same store
        self.noise = Noise(verbose=verbose,
                           filter_data=self._filter_store)

        self.template = Template(verbose=verbose,
                                 filter_data=self._filter_store)

        self.didv = DIDVAnalysis(verbose=verbose,
                                 auto_save_hdf5=auto_save_hdf5,
                                 file_path_name=didv_file_path_name,
                                 filter_data=self._filter_store)

    @property
    def filter_data(self):
        """
        Direct access to the shared filter-data dictionary.
        """
        return self._filter_store

    def describe(self):
        """
        Describe current shared filter data.
        """
        self.store.describe()

    def clear(self, channels=None, tag=None,
              clear_noise_state=True,
              clear_template_state=False,
              clear_didv_state=True):
        """
        Clear shared filter data and optionally processor transient state.
        """
        self.store.clear_data(channels=channels, tag=tag)

        if clear_noise_state:
            self.noise.clear_randoms()

        if clear_template_state:
            if hasattr(self.template, "clear"):
                self.template.clear(channels=channels)

        if clear_didv_state:
            self.didv.clear(channels=channels)

    def load_hdf5(self, file_name, overwrite=True):
        """
        Load filter data into shared store.
        """
        self.store.load_hdf5(file_name, overwrite=overwrite)

    def save_hdf5(self, file_name, overwrite=False):
        """
        Save shared store to file.
        """
        self.store.save_hdf5(file_name, overwrite=overwrite)
