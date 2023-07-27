import warnings
import argparse
from pprint import pprint
from detprocess import TriggerProcessing, FeatureProcessing, Randoms
import os
from pathlib import Path
from pytesdaq.utils import arg_utils


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
    parser.add_argument('--enable-rand', '--enable_rand',
                        dest='enable_rand',
                        action='store_true', help='Acquire randoms')
  #  parser.add_argument('--calc-filter', '--calc_filter',
  #                      dest='calc_filter',
  #                      action='store_true',
  #                      help='Calculate filter informations (PSD and template)')
    parser.add_argument('--enable-trig', '--enable_trig',
                        dest='enable_trig',
                        action='store_true', help='Acquire randoms')
    parser.add_argument('--trigger_dataframe_path', 
                        dest='trigger_dataframe_path', type=str,
                        help='Path to trigger dataframes')
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
    parser.add_argument('--nrandoms', type=int,
                        help='Number random events (either "random_rate" "nrandoms"'
                        + ' required is randoms enabled!)')
    parser.add_argument('--ntriggers', type=int,
                        help='Number trigger events [Default=all available]')
    parser.add_argument('--save_path', type=str,
                        help=('Base path to save process data, '
                              + '(optional)'))
    parser.add_argument('--ncores', type=int,
                        help=('Maximum number of cores used for '
                              'trigger processing [Default=1]'))
    parser.add_argument('--processing_id', type=str,
                        help=('Processing id (short string with no space)'))
    args = parser.parse_args()
    
   
    # ---------------------------
    # Processing type(s)
    # ---------------------------
    acquire_rand = False
    acquire_trig = False
    calc_filter = False
    process_feature = False

    if args.enable_rand:
        acquire_rand = True
    if args.enable_trig:
        acquire_trig = True
    #if args.calc_filter:
    #    calc_filter = True
    if args.enable_feature:
        process_feature = True
    if not (acquire_rand
            or acquire_trig
            or calc_filter
            or process_feature):
        print('An action is required!')
        print('("enable_rand", "calc_filter", "enable_trig",'
              + ' and/or "enable_feature)')
        exit()


    # processing id
    processing_id = None
    if args.processing_id:
        processing_id = args.processing_id

    # nb cores
    ncores = 1
    if args.ncores:
        ncores = int(args.ncores)
        
    # dataframe path
    dataframe_path = None
    if args.trigger_dataframe_path:
        dataframe_path = args.trigger_dataframe_path

    # series
    series = None
    if args.input_series:
        series = arg_utils.extract_list(args.input_series)

        
    # save path
    save_path = None
    if args.save_path:
        save_path = args.save_path

        
         
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
        
    # FIXME, check some fields
    
            
    # ------------------
    # 1. Acquire randoms
    # ------------------

    trigger_output_path = None
    trigger_output_name = None
    
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
        myproc = Randoms(args.input_group_path,
                         processing_id=processing_id,
                         series=series,
                         verbose=True)
        
        myproc.process(random_rate=random_rate,
                       nrandoms=nrandoms,
                       ncores=ncores,
                       lgc_save=True,
                       lgc_output=False,
                       save_path=save_path)

        trigger_output_path = myproc.get_output_path()
        trigger_output_name = str(Path(trigger_output_path).name)
        
    # ------------------
    # 2. Calc Filter
    # ------------------
    if calc_filter:
        print('CALC FILTER NOT AVAILABLE')
        
        
    # ------------------
    # 3. Acquire trigger
    # ------------------
    
    if acquire_trig:

        print('\n\n================================')
        print('Trigger Acquisition')
        print('================================')

        
        ntriggers = -1
        if args.ntriggers:
            ntriggers = int(args.ntriggers)
            
            
        myproc = TriggerProcessing(args.input_group_path,
                                   processing_setup,
                                   series=series,
                                   processing_id=processing_id,
                                   verbose=True)
        
        myproc.process(ntriggers=ntriggers,
                       lgc_output=False,
                       lgc_save=True,
                       output_group_name=trigger_output_name,
                       ncores=ncores,
                       save_path=save_path)
        
        
        dataframe_path = myproc.get_output_path()

        
    # ------------------
    # 4. Process feature
    # ------------------
     
    if process_feature:

        print('\n\n================================')
        print('Feature Processing')
        print('================================')
        
        myproc = FeatureProcessing(args.input_group_path,
                                   processing_setup,
                                   series=series, 
                                   trigger_dataframe_path=dataframe_path,
                                   external_file=None, 
                                   processing_id=processing_id)

        myproc.process(nevents=-1,
                       lgc_save=True,
                       lgc_output=False,
                       ncores=ncores,
                       save_path=save_path)
