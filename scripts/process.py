import warnings
import argparse
from pprint import pprint
from detprocess import utils, TriggerProcessing, IVSweepProcessing, FeatureProcessing, Randoms, Salting
import os
from pathlib import Path
from pytesdaq.utils import arg_utils
from datetime import datetime
import yaml


warnings.filterwarnings('ignore')


if __name__ == "__main__":

    # ------------------
    # Input arguments
    # ------------------
    parser = argparse.ArgumentParser(description='Launch Raw Data Processing')
    parser.add_argument('--raw_path','--input_group_path',
                        dest='input_group_path', type=str, required=True,
                        help='Path to continuous raw data group')
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
                        help='Path to trigger dataframe (threshold and/or randoms)')
    parser.add_argument('--enable-feature', '--enable_feature',
                        dest='enable_feature',
                        action='store_true',
                        help='Process features')
    
    parser.add_argument('-s', '--series', '--input_series',
                        dest='input_series', nargs='+', type=str, 
                        help=('Continous data series name(s) '
                              '(format string Ix_Dyyyymmdd_Thhmmss,'
                              'space or comma seperated) [Default: all series]'))
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
        print('("enable_ivsweep", "enable_rand", "calc_filter", "enable_trig",'
              + ' and/or "enable_feature)')
        exit()

    if (process_ivsweep and
        (acquire_rand  or acquire_trig or  process_feature)):

        print('ERROR: IV sweep processing can only be done without other '
              'type of processing (trigger or feature processing)')
        exit()

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
              
        
    # threshold trigger dataframe path
    trigger_dataframe_path = None
    if args.trigger_dataframe_path:
        trigger_dataframe_path = args.trigger_dataframe_path
  
    # series
    series = None
    if args.input_series:
        series = arg_utils.extract_list(args.input_series)

        
    # save path
    save_path = None
    if args.save_path:
        save_path = str(args.save_path)
         
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
    if (trigger_dataframe_path is not None 
        and  (acquire_trig or acquire_rand)):
        print('ERROR: A trigger dataframe has been provided.'
              'Cannot acquire triggers or randoms. Change arguments!')
        exit()
        
    if (acquire_salting and process_feature
        and not acquire_trig):
        print('ERROR: if salting enabled, trigger acquisition needs to be '
              'done before feature processing!')
        exit()
        
        
    
    # ====================================
    # Calc Filter
    # ====================================
    if calc_filter:
        print('CALC FILTER NOT AVAILABLE')
        



    # ====================================
    # SALTING
    # ====================================

    salting_dataframe_list = [None]
    temp_salting_dataframe_list = [None]
    salting_energy_list = []
    
    if acquire_salting:

        # initialize
        salting_dataframe_list = []

        # FIXME: load yaml file
        yaml_dict = yaml.load(open(processing_setup, 'r'), Loader=utils._UniqueKeyLoader)
        salting_dict = yaml_dict['salting']
        filter_file = yaml_dict['filter_file']
        didv_file = None
        if didv_file in yaml_dict:
            didv_file = yaml_dict['didv_file']
        
        # FIXME: raw data metadata, need to include
        # number of events, trace length, 
        metadata = {''}

        # Number of salt per energy/pdf
        if 'nsalt' in salting_dict:
            nsalt = salting_dict['nsalt']
            salting_dict.pop('nsalt')

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
        salting.set_raw_data_path(group_path=args.input_group_path,
                                  series=series,
                                  restricted=restricted)        
        # loop energies
        for energy in energies:
            temp_salting_dataframe_list = []
            if energy is None:
                print(f'INFO: Generating salting with DM PDF {pdf_file}')
            else:
                print(f'INFO: Generating salting with energy = {energy} eV')
                energy = float(energy)
                
            # store for display
            salting_energy_list.append(energy)
            ntypes = len(salting_dict.items()) #if there are multiple injects for single energy
            for chan, chan_config in salting_dict.items():
                template_tag = chan_config['template_tag']
                noise_tag = chan_config['noise_tag']
                pce = chan_config['collection_efficiency']
                dpdi_tag = chan_config['dpdi_tag']
                dpdi_poles = chan_config['dpdi_poles']

                # generate salt            
                salting.generate_salt(chan,
                                      noise_tag=noise_tag,
                                      template_tag=template_tag,
                                      dpdi_tag=dpdi_tag,
                                      dpdi_poles=dpdi_poles,
                                      energies=energy,
                                      pdf_file=pdf_file,
                                      PCE=pce,
                                      nevents=nsalt,
                                      ntypes = ntypes)
                salting_dataframe = salting.get_dataframe()
                temp_salting_dataframe_list.append(salting_dataframe)
                if ntypes == 1:
                    salting_dataframe_list.append(salting_dataframe)
                if len(temp_salting_dataframe_list) == ntypes > 1: #to handle when multiple chans need same energy
                    salting_dataframe = salting.merge_dataframe(temp_salting_dataframe_list)
                    salting_dataframe_list.append(salting_dataframe)


    
                
    # ====================================
    # Acquire randoms
    # ====================================

    # intialize ouput path list
    randoms_group_path_list = []
    
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


        # generate randoms for each salting
        # FIXME: generate just once in the future
        for salting_df in salting_dataframe_list:
            
            # instantiate
            myproc = Randoms(args.input_group_path,
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

            randoms_group_path_list.append(myproc.get_output_path())
            
                   
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


        for idx, salting_df in enumerate(salting_dataframe_list):

            # display salting energy 
            if salting_energy_list:
                energy = salting_energy_list[idx]
                if energy is not None:
                    print(f'INFO: Processing trigger for salting '
                          f'energy = {energy} eV!')
                    

            # output group name, if randoms already use same group
            # name
            trigger_group_name = None
            if randoms_group_path_list:
                trigger_group_name = str(
                    Path(randoms_group_path_list[idx]).name
                )

            # instantiate
            myproc = TriggerProcessing(args.input_group_path,
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
            
    # ------------------
    # Process feature
    # ------------------
     
    if process_feature:

        print('\n\n================================')
        print('Feature Processing')
        print('================================')

        # check if trigger path exist
        if not trigger_group_path_list:
            trigger_group_path_list = [trigger_dataframe_path]

        
        # loop salting
        for idx, salting_df in enumerate(salting_dataframe_list):

            # display salting energy 
            if salting_energy_list:
                energy = salting_energy_list[idx]
                if energy is not None:
                    print(f'INFO: Processing feature for salting '
                          f'energy = {energy} eV!')
                            
            # trigger dataframes path
            trigger_path = trigger_group_path_list[idx]

            # instantiate
            myproc = FeatureProcessing(args.input_group_path,
                                       processing_setup,
                                       series=series, 
                                       trigger_dataframe_path=trigger_path,
                                       external_file=None, 
                                       processing_id=processing_id,
                                       restricted=restricted,
                                       calib=calib,
                                       salting_dataframe=salting_df)
        
            myproc.process(nevents=-1,
                           lgc_save=True,
                           lgc_output=False,
                           ncores=ncores,
                           save_path=save_path)
            


    # ------------------
    # IV - dIdV sweep
    # processing
    # ------------------

    if process_ivsweep:


        print('\n\n================================')
        print('IV/dIdV Sweep Processing')
        print('================================')
        print(str(datetime.now()))
        
        myproc = IVSweepProcessing(args.input_group_path)
        df = myproc.process(ncores=ncores, lgc_save=True, save_path=save_path)

        



    # done
    print('Processing done! ' + str(datetime.now()))
