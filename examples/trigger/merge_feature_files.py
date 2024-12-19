import detprocess
import numpy as np
import pandas as pd
import vaex
from matplotlib import pyplot as plt
from detanalysis import Analyzer, Semiautocut, ScatterPlotter 
import matplotlib as mpl
import datetime
import sys

vaex.cache.memory()
vaex.settings.cache.memory_size_limit= ('1GB')



# Speed up the process by only checking a few columns
# (see resolve_duplicates function)
columns_to_check = \
    ['trigger_delta_chi2_BigFinsSingleL',
     'trigger_index_BigFinsSingleL',
     'trigger_amplitude_BigFinsSingleL',
     'trigger_delta_chi2_BigFinsSingleR',
     'trigger_index_BigFinsSingleR',
     'trigger_amplitude_BigFinsSingleR',
     'trigger_delta_chi2_BigFinsShared',
     'trigger_index_BigFinsShared',
     'trigger_amplitude_BigFinsShared'
    ]



def resolve_duplicates(group):
    """
    Resolve duplicates in a group of triggers by selecting the row
    with the highest trigger_delta_chi2
    
    Args:
        group (pd.DataFrame): A group of triggers separated by less
            than the merge_window (in samples)
            
    Returns:
        pd.DataFrame: A single row with the highest trigger_delta_chi2.
            This row also contains trigger data from other rows in the
            group.
    """
    
    # Number of triggers in the group
    n_in_group = len(group)

    # If there is only one trigger in the group, return it
    if n_in_group == 1:
        return group

    else:
        # Select the row with the highest trigger_delta_chi2
        best_row_idx = np.argmax(group['trigger_delta_chi2'].values)
        best_row = group[best_row_idx:best_row_idx + 1]

        # Combine non-NaN columns from other rows in the group,
        # which represent trigger data from the non-chosen triggers.
        # This is slow, so by default, we only check a few columns,
        # defined at the top of the script. Altrnatively, we can read all
        # the trigger columns by uncommenting the line below.
                
        # columns_to_check = [c for c in group.columns if 'trigger' in c]
        for col in columns_to_check:
            if best_row[col].values[0] is None or best_row[col].isna().values[0]:
                # For columns where the best row has NaN or the column doesn't 
                # exist, look for non-NaN values in the rest of the group
                for idx in range(n_in_group):
                    row = group[idx:idx+1]
                    if idx != best_row_idx and (not row[col].isna().values[0]) and (row[col].values[0] is not None):
                        best_row[col] = row[col].values

        return best_row



def merge_vaex_dataframes(dfs, merge_window, rowmin=None, rowmax=None):
    """
    Merge multiple Vaex dataframes into one by grouping triggers
    
    Arguments:
        dfs (list): A list of Vaex dataframes to merge
        merge_window (int): The maximum number of samples between triggers
            to consider them as part of the same group
        rowmin (int): Optional, the first row to consider in the merge
        rowmax (int): Optional, the last row to consider in the merge
        
    Returns:
        vaex.dataframe.DataFrame: A single Vaex dataframe with the merged triggers
    
    """
    print(f'Starting to merge\t{datetime.datetime.now()}')

    # Combine all dataframes into one
    combined_df = vaex.concat(dfs)
    nrows = len(combined_df)
    print(f'Concatenated\t{datetime.datetime.now()}')
    
    # Sort by spicestamp for easier processing
    combined_df = combined_df.sort(by="spicestamp")
    print(f'Sorted\t{datetime.datetime.now()}')

    # Group by spicestamp within the merge_window
    groups = []
    current_group = None
    prev_stamp = None

    # Convert to Pandas for faster processing
    if rowmin is not None and rowmax is not None:
        combined_df = combined_df[rowmin:rowmax].to_pandas_df()
    else:
        combined_df = combined_df.to_pandas_df()

    print(f'Converted to Pandas\t{datetime.datetime.now()}')
    
    # Iterate over the rows and group triggers within the merge_window
    spicestamps = combined_df['spicestamp'].values
    for irow, spicestamp in enumerate(spicestamps):

        if irow % 10000 == 0:
            print(f'{irow} of {nrows} rows\t{datetime.datetime.now()}')

        row = combined_df[irow : irow + 1]
        if prev_stamp is None or (spicestamp - prev_stamp) > merge_window:
            if current_group is not None:
                groups.append(current_group)
            current_group = row
        else:
            current_group = pd.concat([current_group, row])

        prev_stamp = spicestamp

    ngroups = len(groups)
    print(f'Groups of triggers found\t{datetime.datetime.now()}')
    
    # Resolve duplicates and merge the results
    resolved_dfs = [None for _ in groups]
    for igroup, group in enumerate(groups):
        resolved_dfs[igroup] = resolve_duplicates(group)
        if igroup % 10000 == 0:
            print(f'{igroup} of {ngroups} groups\t{datetime.datetime.now()}')

    print(f'Coincidences resolved\t{datetime.datetime.now()}')
        
    # Concatenate all resolved groups
    final_df_pd = pd.concat(resolved_dfs)
    print(f'Concatenated groups\t{datetime.datetime.now()}')

    # Convert back to Vaex and return
    final_df = vaex.from_pandas(final_df_pd)
    print(f'Converted dataframe back to Vaex\t{datetime.datetime.now()}')

    return final_df



print(f'Starting \t{datetime.datetime.now()}')

# Load the dataframes
df_right = Analyzer('/home/vvelan/OF_Development/detprocess/scripts/full_singleR/continuous_I2_D20240723_T102114/feature_I2_D20241205_T170928/', series=None).df
df_left = Analyzer('/home/vvelan/OF_Development/detprocess/scripts/full_singleL/continuous_I2_D20240723_T102114/feature_I2_D20241205_T001259/', series=None).df
df_shared = Analyzer('/home/vvelan/OF_Development/detprocess/scripts/full_shared/continuous_I2_D20240723_T102114/feature_I2_D20241204_T224823/', series=None).df

# Note: we could also load the dataframes as below
# df_right = vaex.open(filename.hdf5)
# df_left = vaex.open(filename.hdf5)
# df_shared = vaex.open(filename.hdf5)

# Optionally merge only a subset of the data
if len(sys.argv) == 3:
    rowmin = int(sys.argv[1])
    rowmax = int(sys.argv[2])
else:
    rowmin = None
    rowmax = None

# Calculate "spicestamp", a unique identifier for each trigger timestamp
# which is a combination of series number, dump number, event_number,
# and trigger index
for df in [df_left, df_right, df_shared]:
    df['spicestamp'] = \
        (df['series_number']  - 220240720000000) * 100_000_000_000 + \
        (df['event_number'] // 100000) * 10_000_000_000 + \
        (df['event_number'] % 100000) * 100_000_000 + \
        df['trigger_index']

print(f'Dataframes loaded \t{datetime.datetime.now()}')

# Merge the dataframes
merge_window = 313 # Example merge window, 313 samples = 0.25 ms
final_df = merge_vaex_dataframes([df_left, df_right, df_shared], merge_window, rowmin, rowmax)
print(f'Dataframes merged\t{datetime.datetime.now()}')

# Missing data stored as None/nan which causes the column to be object-type.
# Revert to float.
for k in final_df.get_column_names():
    if final_df[k].dtype == object:
        final_df[k] = final_df[k].astype('float')
print(f'Nones and nans converted\t{datetime.datetime.now()}')

# Save the final dataframe
final_df.export_hdf5(f'merged_features.hdf5')

print(f'Done\t{datetime.datetime.now()}')
