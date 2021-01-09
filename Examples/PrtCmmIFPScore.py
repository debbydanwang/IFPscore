#%reset -f
## State the path for IFPscore if needed. Replace /IFPscore to /Path-to-IFPscore/IFPscore
#import sys
#if '/IFPscore' not in sys.path:
#    sys.path.append('/IFPscore')


import os
import shutil 
import time
import numpy as np
from featurize_pdbbind import load_pdbdata, create_model, validate_model

###############################################################################################################
# example codes                                                                                               #
###############################################################################################################

################################################################################################################
## 1. run PrtCmm IFP Score                                                                                     #
################################################################################################################
np.random.seed(123)
dt_dir = "/Score" # Replace this with the folder where the data folders are  (e.g. 'PDBbind_refined', 'PDBbind_core', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3') 
para = {'alg_type': 'classic', # 'avg', 'classic', 'quasi' or 'qf_pc'
        'ch_para': {'weighted': 0, 'alpha': [-1, -1], 'alpha_step': [0.1, 0.1]},
        'contact_para': {'int_cutoff': 4.5, 'bins': [(0, 4.5)]}, 
        'ifp_para': {'ifptype': 'ecfp', 'degrees': [1, 1], 
                     'prop': ['AtomicNumber', 'TotalConnections', 'HeavyNeighborCount', 'HCount', 'FormalCharge'], 
                     'heavy_atoms': 1, 'hash_type': 'vec', 'idf_power': 64},
        'folding_para': {'power': [7, 8, 9], 'counts': 1},
        'ml_model': 'rf'}


######################################## constructing features ################################################
train_dataset, dt = load_pdbdata(data_dir = dt_dir,
                                 subset = "PDBbind_refined",
                                 train_randsplit_ratio = 1,
                                 split_seed = None,
                                 para = para)
test_dataset_cs, dt = load_pdbdata(data_dir = dt_dir,
                                    subset = "PDBbind_core",
                                    train_randsplit_ratio = 1,
                                    split_seed = None,
                                    para = para)
test_dataset_csar1, dt = load_pdbdata(data_dir = dt_dir,
                                     subset = "PDBbind_csarhiqS1",
                                     train_randsplit_ratio = 1,
                                     split_seed = None,
                                     para = para)
test_dataset_csar2, dt = load_pdbdata(data_dir = dt_dir,
                                     subset = "PDBbind_csarhiqS2",
                                     train_randsplit_ratio = 1,
                                     split_seed = None,
                                     para = para)
test_dataset_csar3, dt = load_pdbdata(data_dir = dt_dir,
                                     subset = "PDBbind_csarhiqS3",
                                     train_randsplit_ratio = 1,
                                     split_seed = None,
                                     para = para)
time.sleep(60)
########################### Buid IFP Score and evaluate it on the validation sets #############################
for pr in para['folding_para']['power']:
    print('\nIFP POWER: ' + str(pr) + '\n')
    print('Constructing PrtCmm IFP Score based on ' + para['ml_model'] + '.............................')
    model = create_model(tp = para['ml_model'])
    # Fit trained model
    print("Fitting model on train dataset.............................")
    model.fit(train_dataset[pr])
    print("Evaluating model")
    r1 = validate_model(mdl = model, vset = test_dataset_cs[pr])
    r2 = validate_model(mdl = model, vset = test_dataset_csar1[pr])
    r3 = validate_model(mdl = model, vset = test_dataset_csar2[pr])
    r4 = validate_model(mdl = model, vset = test_dataset_csar3[pr])
    
    time.sleep(60)
    
# remove tmperary files
drs = ['/tmp/' + fd for fd in os.listdir('/tmp') if 'tmp' in fd]
for dr in drs:
    if os.path.isdir(dr):
        shutil.rmtree(dr, ignore_errors = True)

