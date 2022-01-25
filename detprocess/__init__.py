from . import process
from .process import process_data, SingleChannelExtractors
from . import io
from .core import *

# load seaborn colormaps
from seaborn import cm
del cm
