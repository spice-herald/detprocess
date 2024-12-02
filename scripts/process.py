import warnings
import argparse
from pprint import pprint
from detprocess import utils, TriggerProcessing, IVSweepProcessing
from detprocess import  FeatureProcessing, Randoms, Salting, RawData
import os
from pathlib import Path
from pytesdaq.utils import arg_utils
from datetime import datetime
import yaml
import vaex as vx
import re
from qetpy.utils import convert_channel_name_to_list,convert_channel_list_to_name
import gc
import multiprocessing
import numpy as np
import subprocess
import threading

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# script path
script_path = os.path.abspath(__file__)


# subprocess output
def stream_output(stream, prefix):
    """
    Read a stream line-by-line and print it with a prefix.
    """
    for line in iter(stream.readline, ''):
        print(f'[{prefix}]: {line.strip()}', flush=True)
        
        

if __name__ == "__main__":

    # multi processing
    multiprocessing.set_start_method('spawn')

    # vaex
    vx.multithreading.thread_count = 1

    # script path
    script_path = os.path.abspath(__file__)
    
    # ------------------
    # Input arguments
    # ------------------
    parser = argparse.ArgumentParser(description='Launch Raw Data Processing')

    parser.add_argument('--raw_path','--input_group_path',
                        dest='raw_group_path', type=str, required=True,
                        help='Path to continuous raw data group')

    parser.add_argument('-s', '--series', '--input_series',
                        dest='raw_series', nargs='+', type=str, 
                        help=('Continous data series name(s) '
                              '(format string Ix_Dyyyymmdd_Thhmmss,'
                              'space or comma seperated) [Default: all series]'))
    
    parser.add_argument('--processing_setup', type=str,
                        help='Processing setup file')        

    parser.add_argument('--enable-rand', '--enable-randoms', '--enable_rand',
                        dest='enable_rand',
                        action='store_true', help='Acquire randoms')
    #  parser.add_argument('--calc-filter', '--calc_filter',
    #                      dest='calc_filter',
    #                      action='store_true',
    #                      help='Calculate filter informations (PSD and template)')

    parser.add_argument('--enable-ivsweep', dest='enable_ivsweep',
                        action='store_true', help='Process IV sweep data')

    parser.add_argument('--enable-trig', '--enable-triggers', '--enable_trig',
                        dest='enable_trig',
                        action='store_true', help='Acquire triggers')

    parser.add_argument('--enable-salting', '--enable_salting',
                        dest='enable_salting',
                        action='store_true', help='Generate and inject salting')

    parser.add_argument('--trigger_dataframe_path',
                        dest='trigger_dataframe_path', type=str,
                        help='Path trigger dataframe (threshold and/or randoms) ')

    parser.add_argument('--trigger_series',
                        dest='trigger_series', nargs='+', type=str, 
                        help=('Trigger series '
                              '(format string Ix_Dyyyymmdd_Thhmmss,'
                              'space or comma seperated) [Default: all series]'))
    
    parser.add_argument('--salting_dataframe_path', 
                        dest='salting_dataframe_path', type=str,
                        help='Path to salting dataframe')

    parser.add_argument('--enable-feature', '--enable_feature',
                        dest='enable_feature',
                        action='store_true',
                        help='Process features')

    parser.add_argument('--random_rate', type=float,
                        help='random event rate (either "random_rate" "nrandoms"'
                        + ' required is randoms enabled!)')

    parser.add_argument('--nrandoms', type=int, dest='nrandoms',
                        help='Number random events (either "random_rate" "nrandoms"'
                        + ' required is randoms enabled!)')

    parser.add_argument('--ntriggers', type=int, dest='ntriggers',
                        help='Number trigger events [Default=all available]')

    parser.add_argument('--save_path', type=str,
                        help=('Base path to save process data, '
                              + '(optional)'))

    parser.add_argument('--ncores', type=int,
                        help=('Maximum number of cores used for '
                              'trigger processing [Default=1]'))
    
    parser.add_argument('--processing_id', type=str,
                        help=('Processing id (short string with no space)'))

    parser.add_argument('--restricted',
                        action='store_true',
                        help=('Process restricted (blinded) data'))

    parser.add_argument('--calib',
                        action='store_true',
                        help=('Processing calibration data'))


    parser.add_argument('--feature_series_name', type=str,
                        help='Feature processing output series name')

    parser.add_argument('--use_subprocess',
                        action='store_true',
                        help='Launch with subprocess')
    
    parser.add_argument('--subprocess_num', type=int,
                        help='Subprocess number')

    
    
    

    
    args = parser.parse_args()
    
   
    # ---------------------------
    # Processing type(s)
    # ---------------------------
    process_ivsweep = False
    acquire_salting = False
    acquire_rand = False
    acquire_trig = False
    calc_filter = False
    process_feature = False
    use_subprocess = False
    
    if args.enable_ivsweep:
        process_ivsweep = True
    if (args.enable_rand or args.random_rate or args.nrandoms):
        acquire_rand = True
    if args.enable_trig:
        acquire_trig = True
    #if args.calc_filter:
    #    calc_filter = True
    if args.enable_salting:
        acquire_salting = args.enable_salting

    if args.enable_feature:
        process_feature = True
    if not (process_ivsweep
            or acquire_salting
            or acquire_rand
            or acquire_trig
            or calc_filter
            or process_feature):
        print('An action is required!')
        print('("enable_ivsweep", "enable_rand", "calc_filter", "enable_trig", '
              + '"enable_salting" and/or "enable_feature)')
        exit()

    if (process_ivsweep and
        (acquire_rand  or acquire_trig or  process_feature)):

        print('ERROR: IV sweep processing can only be done without other '
              'type of processing (trigger or feature processing)')
        exit()


    if args.use_subprocess:
        use_subprocess = True
        
    subprocess_num = None
    if args.subprocess_num:
        subprocess_num = int(args.subprocess_num)

    feature_series_name = None
    if args.feature_series_name:
        feature_series_name = args.feature_series_name

        
    restricted = False
    if args.restricted:
        restricted = True

    calib = False
    if args.calib:
        calib = True 
        restricted = False
        
    # processing id
    processing_id = None
    if args.processing_id:
        processing_id = args.processing_id

    # nb cores
    ncores = 1
    if args.ncores:
        ncores = int(args.ncores)

    if args.ntriggers and ncores>1:
        print('\nERROR: Multiple cores only possible when '
              'processing all events')
        print('Set argument "--ncores 1" ')
        exit()
        
    # trigger dataframe path
    trigger_dataframe_path = None
    if args.trigger_dataframe_path:
        trigger_dataframe_path = args.trigger_dataframe_path

    # salting  dataframe path
    salting_dataframe_path = None
    if args.salting_dataframe_path:
        salting_dataframe_path = args.salting_dataframe_path
          
    # series
    series = None
    if args.raw_series:
        series = arg_utils.extract_list(args.raw_series)

    # check raw data path
    raw_group_path = args.raw_group_path
    raw_group_path = os.path.normpath(raw_group_path)
    if not os.path.isdir(raw_group_path):
        print(f'ERROR: Raw data directory '
              f'"{raw_group_path} not found!')
        exit()
        
    
    # save path
    save_path = None
    if args.save_path:
        save_path = str(args.save_path)
    if save_path is None:
        save_path = os.path.dirname(raw_group_path) + '/processed'
        if '/raw/processed' in save_path:
            save_path = save_path.replace('/raw/processed', '/processed')

            
    # series
    trigger_series = None
    if args.trigger_series:
        trigger_series = arg_utils.extract_list(args.trigger_series)

            
    # ------------------
    # check setup file
    # ------------------
    processing_setup = None
    if args.processing_setup:
        processing_setup = args.processing_setup
        if not os.path.isfile(processing_setup):
            print('ERROR: Processing setup file "'
                  + processing_setup  + '" not found!')
            exit()

    if (processing_setup is None 
        and (acquire_trig or process_feature)):
        print('ERROR: Processing setup required!')
        exit()
        
    # Check some fields
    if trigger_dataframe_path is not None:
        
        if acquire_trig or acquire_rand:
            print('ERROR: A trigger dataframe has been provided.'
                  'Cannot acquire triggers or randoms. Change arguments!')
            exit()

        if acquire_salting and salting_dataframe_path is None:
            print('ERROR: A trigger dataframe has been provided.'
                  'Cannot regenerate salting dataframe!')
            exit()
            
    if salting_dataframe_path is not None:

        if (trigger_dataframe_path is None and process_feature
             and not acquire_trig):
            print('ERROR: if salting path provided and not trigger path, '
                  'trigger needs to be acquired before feature processing!')
        
    elif (acquire_salting and process_feature
          and not acquire_trig):
        print('ERROR: if salting enabled, trigger acquisition needs to be '
              'done before feature processing!')
        exit()
    
    if acquire_salting and acquire_rand:
        print('ERROR: For the moment, randoms cannot '
              'be enabled in the same time as salting!')
        exit()
        

    # ====================================
    # Check raw data and processing
    # ====================================
    print('Processing information')
    print('======================')
    rawdata = RawData(raw_group_path)
    rawdata.describe()
    group_name = rawdata.get_group_name()
    base_path = rawdata.get_base_path()
    facility = rawdata.get_facility()
    
    if not process_ivsweep:
      
        processing_steps = []
        if acquire_salting and salting_dataframe_path is None:
            processing_steps.append('generate salting')
        if acquire_rand:
            processing_steps.append('acquire randoms')
        if acquire_trig:
            processing_steps.append('acquire triggers')
        if process_feature:
            processing_steps.append('process features')

        process_str = ', '.join(processing_steps)
        print(f'\nThe following processing with done:')
        print(f' - {process_str}')
             
        if acquire_salting or salting_dataframe_path is not None:
            print(' - salting will be injected in raw data')

        if series is not None:
            print(f' - only the following series with be processed: '
                  f'{series}!')

        if restricted:
            print(f' - restricted data will be processed!')
        else:
            print(' - no restricted data will be processed (open only)')


    # ====================================
    # Processing config
    # ====================================
    
    yaml_dict = yaml.load(open(processing_setup, 'r'), Loader=utils._UniqueKeyLoader)
    filter_file = yaml_dict['filter_file']
    didv_file = None
    if didv_file in yaml_dict:
        didv_file = yaml_dict['didv_file']
                
     
    # ====================================
    # Calc Filter
    # ====================================
    if calc_filter:
        print('CALC FILTER NOT AVAILABLE')
        
        


    # ====================================
    # SALTING
    # ====================================

    salting_dataframe_list = [None]
    salting_energy_list = []
    
    if acquire_salting and salting_dataframe_path is None:

        print('\n\n================================')
        print('Salting generation')
        print('================================')
        
        # initialize
        salting_dataframe_list = []

        # build output path 
        output_path = f'{save_path}/{group_name}'
        series_name = utils.create_series_name(facility)
        salting_group_name = f'salting_{series_name}'
        output_path += f'/{salting_group_name}'

        # create directory
        utils.create_directory(output_path)
        
        # get salting dict
        salting_dict = yaml_dict['salting']
                
        # FIXME: raw data metadata, need to include
        # number of events, trace length, 
        metadata = {''}

        # Number of salt per energy/pdf
        if 'nsalt' in salting_dict:
            nsalt = salting_dict['nsalt']
            salting_dict.pop('nsalt')
          
        coincident_salts = False
        if "coincident_salts" in salting_dict:
            coincident_salts = salting_dict['coincident_salts']
            print(f'INFO: Salt time coincidence between channels has been set to {coincident_salts}!')
            salting_dict.pop('coincident_salts')
        
        # DM pdf
        pdf_file = None
        if 'dm_pdf_file' in salting_dict:
            pdf_file = salting_dict['dm_pdf_file']
            salting_dict.pop('dm_pdf_file')

        # if "energies" provided, use instead of pdf
        energies = [None]
        if 'energies' in salting_dict:
            energies = salting_dict['energies']
            salting_dict.pop('energies')
            if not isinstance(energies, list):
                energies = [energies]
            if pdf_file is not None:
                print('ERROR with salting parameters: Either energies or pdf '
                      'should be provided. Not both!')
                exit(0)
                
        
        # Instantiate salting
        salting = Salting(filter_file, didv_file=didv_file)

        # Add either raw data metadata or raw path
       
        #salting.set_raw_data_metadata(...)
        salting.set_raw_data_path(group_path=raw_group_path,
                                  series=series,
                                  restricted=restricted)        
        # loop energies
        for energy in energies:

            # save energy (for display later)
            salting_energy_list.append(energy)
                    
            # display
            if energy is None:
                print(f'INFO: Generating salting with DM PDF {pdf_file}')
            else:
                print(f'INFO: Generating salting with energy = {energy} eV')
                energy = float(energy)

            # intialize dataframe list for channel
            chan_dataframe_list = []
            i = 0
            for chan, chan_config in salting_dict.items():

                # check if multi-channel
                chan_list = convert_channel_name_to_list(chan)
                
                # get config
                template_tag = chan_config['template_tag']
                noise_tag = chan_config['noise_tag']
                dpdi_tag = chan_config['dpdi_tag']
                dpdi_poles = chan_config['dpdi_poles']

                pce = 1
                if 'collection_efficiency' in chan_config:
                    pce = chan_config['collection_efficiency']
                elif len(chan_list) >=2:
                    pce = [pce]*len(chan_list)

                coinchan = False
                if coincident_salts is True and i > 0:
                    coincidenttimes_dataframe = salting.get_injectiontimes()
                    salting.set_dataframe(coincidenttimes_dataframe)
                # generate salt
                salting.generate_salt(chan,
                                      noise_tag=noise_tag,
                                      template_tag=template_tag,
                                      dpdi_tag=dpdi_tag,
                                      dpdi_poles=dpdi_poles,
                                      energies=energy,
                                      pdf_file=pdf_file,
                                      PCE=pce,
                                      nevents=nsalt)
                
                
                salting_dataframe = salting.get_dataframe()
                chan_dataframe_list.append(salting_dataframe)
                i += 1
                salting.clear_dataframe()

            # concatanate if needed
            final_dataframe = chan_dataframe_list[0]
            if len(chan_dataframe_list) > 1:
                final_dataframe = vx.concat(chan_dataframe_list)
         
            # save to hdf5 
            series_name = utils.create_series_name(facility)
            file_name = f'salting_pdf_{series_name}_F0001.hdf5'
            if energy is not None:
                file_name = f'salting_{energy}eV_{series_name}_F0001.hdf5'
            if processing_id is not None:
                file_name = f'{processing_id}_{file_name}'
            salting_file_path = f'{output_path}/{file_name}'
            final_dataframe.export_hdf5(salting_file_path, mode='w')
                      
            # append
            salting_dataframe_list.append(salting_file_path)


            # cleanup
            del final_dataframe

        # cleanup
        del salting
        gc.collect()  # Force garbage collection


    # case salting dataframe path provided
    if salting_dataframe_path is not None:
        salting_dataframe_list = [salting_dataframe_path]
        print(f' - salting dataframe provided: '
              f'{salting_dataframe_path}')

        
    # ====================================
    # Acquire randoms
    # ====================================

    # intialize ouput path list
    randoms_group_path = None
    
    if acquire_rand:

             
        print('\n\n================================')
        print('Randoms Acquisition')
        print('================================')

        if args.random_rate and args.nrandoms:
            print('ERROR: Choose between "random_rate" '
                  + 'or "nrandoms" argument, not both!')
            exit()
        
        random_rate = None
        nrandoms = None
        if args.random_rate:
            random_rate = float(args.random_rate)
        elif args.nrandoms:
            nrandoms = int(args.nrandoms)
        else:
            print('ERROR: "random_rate" or '
                  + '"nrandoms" argument required!')
            exit()
            
        # instantiate
        myproc = Randoms(raw_group_path,
                         processing_id=processing_id,
                         series=series,
                         restricted=restricted,
                         calib=calib,
                         verbose=True)
            
        myproc.process(random_rate=random_rate,
                       nrandoms=nrandoms,
                       ncores=ncores,
                       lgc_save=True,
                       lgc_output=False,
                       save_path=save_path)
        
        randoms_group_path = myproc.get_output_path()
        
                   
    # ====================================
    # Acquire trigger
    # ====================================

    # initialize group list
    trigger_group_path_list = []
      
    if acquire_trig:

        print('\n\n================================')
        print('Trigger Acquisition')
        print('================================')

        
        ntriggers = -1
        if args.ntriggers:
            ntriggers = int(args.ntriggers)


        # trigger group name: if no salting done -> save in
        # randoms directory
        trigger_group_name = None
        if (randoms_group_path is not None
            and salting_dataframe_list[0] is None):
            trigger_group_name = os.path.basename(randoms_group_path)
            

        for idx, salting_df in enumerate(salting_dataframe_list):

            # display salting energy 
            if salting_energy_list:
                energy = salting_energy_list[idx]
                if energy is not None:
                    print(f'INFO: Processing trigger with salting '
                          f'energy = {energy} eV!')
                elif salting_df is not None:
                     print(f'INFO: Processing trigger with salting '
                           f'done using DM PDF!')
                    
            # instantiate
            myproc = TriggerProcessing(raw_group_path,
                                       processing_setup,
                                       series=series,
                                       processing_id=processing_id,
                                       restricted=restricted,
                                       calib=calib,
                                       salting_dataframe=salting_df,
                                       verbose=True)
          
            myproc.process(ntriggers=ntriggers,
                           lgc_output=False,
                           lgc_save=True,
                           output_group_name=trigger_group_name,
                           ncores=ncores,
                           save_path=save_path)

            trigger_group_path_list.append(myproc.get_output_path())

            # cleanup
            del myproc
            gc.collect()  # Force garbage collection

    elif trigger_dataframe_path is not None:
        trigger_group_path_list = [trigger_dataframe_path]


        
    # ======================================
    # Process feature
    # ======================================
     
    if process_feature:

        print('\n\n================================')
        print('Feature Processing')
        print('================================')

        # check if trigger path exist
        if not trigger_group_path_list:
            trigger_group_path_list = [randoms_group_path]
    
        # loop salting
        for idx, salting_df in enumerate(salting_dataframe_list):

            # display salting energy 
            if salting_energy_list:
                energy = salting_energy_list[idx]
                if energy is not None:
                    print(f'INFO: Processing feature for salting '
                          f'energy = {energy} eV!')
                elif salting_df is not None:
                    print(f'INFO: Processing trigger with salting '
                          f'done using DM PDF!')
                    
            # trigger dataframes path
            trigger_path = trigger_group_path_list[idx]


            #  subprocess
            if ncores>1 and use_subprocess:
                
                # find trigger series list
                series_list = utils.get_dataframe_series_list(
                    trigger_path
                )
                
                if ncores > len(series_list):
                    print(f'\nWARNING: Changing the number of cores to '
                          f'{nb_series} (maximum allowed)')
                    ncores = nb_series
                series_split_temp = np.array_split(series_list, ncores)
                series_split = []
                for series_sublist in series_split_temp:
                    if series_sublist.size == 0:
                        continue
                    series_sublist = list(series_sublist)
                    series_string = ','.join(series_sublist)
                    series_split.append(series_string)

                # output series
                output_series_name =  utils.create_series_name(facility)
                    
                # launch processes
                processes = []
                threads = []
                counter = 0
                for aseries in series_split:

                    counter += 1
                    
                    # build command line
                    cmd_list = ['python3', '-u', f'{script_path}']
                    cmd_list.extend(['--raw_path', f'{raw_group_path}'])
                    cmd_list.extend(['--trigger_series', f'{aseries}'])
                    cmd_list.extend(['--processing_setup', f'{processing_setup}'])
                    cmd_list.extend(['--enable-feature', '--ncores', '1'])
                    cmd_list.extend(['--trigger_dataframe_path', f'{trigger_path}'])
                    cmd_list.extend(['--feature_series_name', f'{output_series_name}'])
                    cmd_list.extend(['--subprocess_num', f'{counter}'])
                    if processing_id is not None:
                        cmd_list.extend(['--processing_id', f'{processing_id}'])
                    if restricted:
                        cmd_list.append('--restricted')
                    if salting_df is not None:
                        cmd_list.extend(['--salting_dataframe_path', f'{salting_df}'])
                    if calib:
                        cmd_list.append('--calib')
                    if save_path is not None:
                        cmd_list.extend(['--save_path', f'{save_path}'])
                        
                    # launch
                    print(f'INFO: Launching subprocess for series {aseries}!')
                    p = subprocess.Popen(cmd_list,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         text=True,
                                         bufsize=1
                                         )
                    processes.append(p)

                    
                    # Create threads to stream the output
                    stdout_thread = threading.Thread(
                        target=stream_output,
                        args=(p.stdout, f'Subprocess #{counter}: ')
                    )
                    stderr_thread = threading.Thread(
                        target=stream_output,
                        args=(p.stderr, f'Subprocess #{counter} [STDERR]')
                    )

                    stdout_thread.start()
                    stderr_thread.start()

                    threads.append(stdout_thread)
                    threads.append(stderr_thread)
                    
                # Wait for all threads and subprocesses to complete
                for t in threads:
                    t.join()

                for p in processes:
                    p.wait()
                    if p.returncode != 0:
                        raise ValueError(f'ERROR: Subprocess for sfailed with return code {p.returncode}')
                    
                print('INFO: Subprocess completed successfully!')
                                        
            else:
                     
                # instantiate
                myproc = FeatureProcessing(raw_group_path,
                                           processing_setup,
                                           series=series, 
                                           trigger_dataframe_path=trigger_path,
                                           trigger_series=trigger_series,
                                           external_file=None, 
                                           processing_id=processing_id,
                                           restricted=restricted,
                                           calib=calib,
                                           salting_dataframe=salting_df)
        
                myproc.process(nevents=-1,
                               lgc_save=True,
                               lgc_output=False,
                               ncores=ncores,
                               output_series_name=feature_series_name,
                               subprocess_num=subprocess_num,
                               save_path=save_path)
            

                # cleanup
                del myproc
                gc.collect()  # Force garbage collection


                
    # ------------------
    # IV - dIdV sweep
    # processing
    # ------------------

    if process_ivsweep:


        print('\n\n================================')
        print('IV/dIdV Sweep Processing')
        print('================================')
        print(str(datetime.now()))
        
        myproc = IVSweepProcessing(raw_group_path)
        df = myproc.process(ncores=ncores, lgc_save=True,
                            save_path=save_path)

        



    # done
    print('Processing done! ' + str(datetime.now()))
