from loguru import logger
import warnings

warnings.filterwarnings('ignore')


from offlinerl.config.algo import  edac_config, mcq_config, cql_config, plas_config, mopo_config, moose_config, bcqd_config, bcq_config, bc_config, crr_config, combo_config, bremen_config, maple_config, mobile_config, rambo_config, td3bc_config, bc_model_config, maple_config_new,prdc_config
from offlinerl.utils.config import parse_config
from offlinerl.algo.modelfree import cql, plas, bcqd, bcq, bc, crr, edac, mcq, td3bc, prdc
from offlinerl.algo.modelbase import mopo, moose, combo, bremen, maple, mobile, rambo, maple_new
from offlinerl.algo.dynamics_model import bc_model

algo_dict = {
    'edac' : {"algo" : edac, "config" : edac_config},
    'bc' : {"algo" : bc, "config" : bc_config}, #
    'bcq' : {"algo" : bcq, "config" : bcq_config},
    'mcq' : {"algo" : mcq, "config" : mcq_config},
    'bcqd' : {"algo" : bcqd, "config" : bcqd_config},
    'combo' : {"algo" : combo, "config" : combo_config},
    "cql" : {"algo" : cql, "config" : cql_config}, #
    "crr" : {"algo" : crr, "config" : crr_config},
    "plas" : {"algo" : plas, "config" : plas_config},
    "prdc" : {"algo" : prdc, "config" : prdc_config},
    'moose' : {"algo" : moose, "config" : moose_config},
    'mopo': {"algo" : mopo, "config": mopo_config},
    'bremen' : {"algo" : bremen, "config" : bremen_config},
    'maple': {'algo':maple , 'config':maple_config},
    'mobile': {'algo':mobile , 'config':mobile_config},
    'rambo': {'algo':rambo , 'config':rambo_config},
    'td3bc': {'algo':td3bc , 'config':td3bc_config}, #
    'bc_model': {'algo':bc_model , 'config':bc_model_config},
    'maple_new': {'algo':maple_new , 'config':maple_config_new},
}

def algo_select(command_args, algo_config_module=None):
    algo_name = command_args["algo_name"]
    logger.info('Use {} algorithm!', algo_name)
    assert algo_name in algo_dict.keys()
    algo = algo_dict[algo_name]["algo"]
    
    if algo_config_module is None:
        algo_config_module = algo_dict[algo_name]["config"]
    algo_config = parse_config(algo_config_module)
    algo_config.update(command_args)
    
    algo_init = algo.algo_init
    algo_trainer = algo.AlgoTrainer
    
    return algo_init, algo_trainer, algo_config
    
    