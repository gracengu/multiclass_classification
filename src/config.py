import os, inspect
base_path, current_dir =  os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Config(object):

    DATA = dict(
        INPUT_PATH          = os.path.join(base_path ,"./data/data.csv"),
        PROCESS_INPUT_PATH  = os.path.join(base_path ,"../data/treated_transformed_data.csv"),
        PCA_INPUT_PATH      = os.path.join(base_path ,'../models/pca_v1.sav')

    )
        
    ANALYSIS_CONFIG = dict(
        UNIQUE_THRESHOLD            = 20,
        RANDOM_SEED                 = 123,
        K_THRESHOLD                 = 10,
        BITRIMODAL_DISTRIBUTION     = ["105", "145", "147", "82"],
        XCHART_COLUMNS              = ['0', '9', '25', '47', '50' ,'56' ,'64' ,'73', '85', '98' ,'100', '108' ,\
                                        '128' ,'133', '137', '117', '125'],
        TRANSFORMED_COLUMNS         = ['0_log', '9_log', '25_log', '47_log', '50_log' ,'56_log' ,'64' ,'73_log', \
                                        '85_sqrt', '98_log' ,'100_log', '108_log' ,'128_log' ,'133_log','137_log', \
                                        '117_log', '125_log']
    )

    PLOT_CONFIG = dict(
        FIG_SIZE            = (12,8),
        FIGURE_STYLE        = "whitegrid",
        FONT_SIZE           = 20,
    )

    MODELLING_CONFIG = dict(
        RANDOM_SEED         = 123
        
    )
        
    METRICS_THRESHOLD_PLOT = dict(
        
    )