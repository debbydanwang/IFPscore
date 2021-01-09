import os
import time
import logging
import numpy as np
import deepchem as dc
from sklearn.metrics import mean_squared_error
import multiprocessing
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from IFP import pro_lig_ifp


def get_para_for_print(para):
    """
    Get a list of strings that describes the parameters. Returns the string list.
    """
    forprint = ["####################### parameters #################################\n"]
    for (k, v) in para.items():
        if isinstance(v, dict):
            tmpstr = "%s: {" % k
            for (key, val) in v.items():
                tmpstr += "%s: %s; " % (key, val)
            tmpstr += "}\n"
            forprint.append(tmpstr)                
        else:
            forprint.append("%s: %s\n" % (k, v))
    forprint.append("-------------------------------------------------------\n")
    return forprint


def _featurize_complex(pdbinfo,
                       log_message,
                       para):
    """Featurizes a complex.
    First initializes a pro_lig_ifp class (lig_file, pro_file, int_cutoff, alpha and alpha_step),
    then finds the contacting atoms and computes the ifp.
    Returns 1-D features of length 2^power x bins or 2^power x bins x 2 (ifptype = ecfp).
    """   
    logging.info(log_message)
    try:
        PLI = pro_lig_ifp(fn_pro = pdbinfo['pro_file'], 
                          fn_lig = pdbinfo['lig_file'],
                          pdb = pdbinfo['pdbid'],
                          int_cutoff = para['contact_para']['int_cutoff'], 
                          ch_para = para['ch_para'],
                          contact_bins = para['contact_para']['bins'])
    
        PLI.find_contacts()
        if para['alg_type'] in ['avg', 'classic']:
#            return PLI.featurize_ifp(contact_bins = None,
#                                     ifptype = para['ifp_para']['ifptype'],    
#                                     alg_type = para['alg_type'],
#                                     ecfp_radius = para['ifp_para']['degrees'],
#                                     heavy_atoms = para['ifp_para']['heavy_atoms'],
#                                     base_prop = para['ifp_para']['prop'],
#                                     hash_type = para['ifp_para']['hash_type'],
#                                     idf_power = para['ifp_para']['idf_power'],
#                                     folding_para = para['folding_para'])
            return PLI.featurize_ifp_mulpr(contact_bins = None,
                                           ifptype = para['ifp_para']['ifptype'],    
                                           alg_type = para['alg_type'],
                                           ecfp_radius = para['ifp_para']['degrees'],
                                           heavy_atoms = para['ifp_para']['heavy_atoms'],
                                           base_prop = para['ifp_para']['prop'],
                                           hash_type = para['ifp_para']['hash_type'],
                                           idf_power = para['ifp_para']['idf_power'],
                                           folding_para = para['folding_para'])
        elif para['alg_type'] == 'quasi':
            return PLI.get_quasi_fragmental_desp()
        elif para['alg_type'] == 'qf_pc':
            return PLI.get_qf_pc_desp()
        else:
            print('Wrong algorithm type!!!')
            return None
    except:
        return None
    
def featurize_complexes(ligand_files, 
                        protein_files,
                        pdbids,
                        para = {'alg_type': 'classic',
                                'contact_para': {'int_cutoff': 4.5, 'bins': [(0, 4.5)]},
                                'ch_para': {'weighted': 0, 'alpha': [-1, -1], 'alpha_step': [0.1, 0.1]},
                                'ifp_para': {'ifptype': 'ecfp', 'degrees': [2, 2], 
                                             'prop': ['SolidAngle'], 
                                             'heavy_atoms': 0, 'hash_type': 'str', 'idf_power': 64},
                                'folding_para': {'power': np.arange(6, 16, 1), 'counts': 0}}):
    """Featurizes a group of complexes using interaction fingerprints.
    Parameters:
        ligand_files: a list of ligand files (mostly in .mol2 or .sdf)
        protein_files: a list of protein files (mostly in .pdb)
        pdbids: a list of pdb ids for the complexes under processing
        alg_type - ifp algorithm ('avg', 'classic' or 'quasi' - coming soon)
        int_cutoff: distance cutoff for identifying contacting atoms in protein-ligand interfaces
        bins - for finding protein-ligand contacts in different ranges   
        ch_para - parameters for constructing alpha shapes (concave hulls) for protein and ligand 
                weighted: weighted alpha shape (1) or mlnot (0)
                alpha: alpha values for protein (alpha[0]) and ligand (alpha[1])
                alpha_step: steps for tuning alpha shapes if alpha = -1 (alpha_step[0] for protein and alpha_step[1] fro ligand)
        ifptype - interaction fingerprint type, 'splif', 'ecfp' or 'plec'
        degrees - ECFP radii for the ifp (e.g. [2] for 'ecfp' and 'splif', [3, 1] for 'plec')
        prop - a list of atomic properties, full list as below. NOTE the fast version only needs one base property (prop[0])
               'AtomicNumber': the atomic number of atom
               'TotalConnections': the degree of the atom in the molecule including Hs
               'HeavyNeighborCount': the number of heavy (non-hydrogen) neighbor atoms
               'HCount': the number of attached hydrogens (both implicit and explicit)
               'FormalCharge': the formal charge of atom
               'DeltaMass': the difference between atomic mass and atomic weight (weighted average of atomic masses)
               'IsRingAtom': indicates whether the atom is part of at least one ring
               'Aromaticity': indicates whether the atom is aromatic
               'IsTerminalAtom': indicates whether the atom is a terminal atom
               'SolidAngle': the solid angle of the atom on the molecule surface (> 0: convex, < 0: concave)
               'SolidAngleSign': the sign of solid angle of the atom (-1, 0, 1)
        heavy_atoms: use heavy atoms or all atoms
        hash_type: type for the hash function ('str' or 'vec')
        idf_power: power for the identifiers hashed by hash_ecfp ('str')
        folder_para - parameters for fingerprint folding
                power: fingerprint folding size (2^power bits)
                counts: use occurences of identifiers (1) or not (0)        
    Returns an array of computed features (n' x m, where n' = n - f and m = 2^power x bins or 2^power x bins x 2) 
    and an index list of failed complexes (length of f)
    """
    pool = multiprocessing.Pool(processes = 15)
    features = {}
    results = []
    for i, (lig_file, pro_file, pdbid) in enumerate(zip(ligand_files, protein_files, pdbids)):
        log_message = "Featurizing %d / %d" % (i, len(lig_file))
        pdbinfo = {'pro_file': pro_file, 'lig_file': lig_file, 'pdbid': pdbid}
        results.append(pool.apply_async(_featurize_complex, (pdbinfo, log_message, para)))      
    pool.close()
    feat = []
    failures = []
    for ind, result in enumerate(results):
        new_features = result.get()
        if new_features is None:
            failures.append(ind)
        else:            
            feat.append(new_features)
    
    if para['alg_type'] in ['quasi', 'qf_pc']:
        return np.vstack(feat), failures
    else:
        for pr in para['folding_para']['power']:  
            tmp = [r[pr] for r in feat]
            features[pr] = np.vstack(tmp)
        return features, failures


def load_pdbdata(data_dir = None,
                 subset = "PDBbind_core",
                 ligfiletype = '.pdb',
                 train_randsplit_ratio = 1,
                 cv_folds = 3,
                 para = {'alg_type': 'classic',
                         'contact_para': {'int_cutoff': 4.5, 'bins': [(0, 4.5)]},
                         'ch_para': {'weighted': 0, 'alpha': [-1, -1], 'alpha_step': [0.1, 0.1]},
                         'ifp_para': {'ifptype': 'ecfp', 'degrees': [2, 2],
                                      'prop': ['SolidAngle'], 
                                      'heavy_atoms': 0, 'hash_type': 'str', 'idf_power': 64},
                         'folding_para': {'power': np.arange(6, 16, 1), 'counts': 0}},                 
                 split_seed = None):
    """Load PDBbind data set (version 2019).
    Parameters:
        data_dir - folder that stores PDBbind data
        subset - 'PDBbind_refined', 'PDBbind_core', 'csarhiqS1', 'csarhiqS2' or 'csarhiqS3'; indicating the refined, core set of PDBbind 2019 or csar-hiq sets 1~3
        ligfiletype - file type for the ligand files: '.pdb' or '.lig' (.pdb is used as default)
        train_randsplit_ratio: ratio for random splits (1 - no splits, 0 - perform cross-valiation splits, (0, 1) - raio for training set and rest for testing)
        cv_folds - cross-valistion folds (if train_randsplit_ratio = 0)
        para - a dictionary of parameters for computing ifp features (align with featurize_complexes function)
        split_seed - random seed for splits
    Returns (training data, test data) or a cross-validation data set, from the deepchem library
    """
    datasets = {}
    all_datasets = {}
    
    if data_dir == None:
        print("\nData directory is missing!\n")
    else:
        print("\nWorking directory is %s\n" % data_dir)
    os.chdir(data_dir)
    print("\nRaw dataset in %s\n" % subset)
    
    # read in pdb indexes, binding affinity (RT log(Kd/Ki))
    # extract location of data (protein, ligand)
    if "core" in subset:
        index_labels_file = os.path.join("indexes", "cs.txt")
    elif "refined" in subset:
        index_labels_file = os.path.join("indexes", "rs-cs-csar.txt")
    elif "csarhiqS1" in subset:
        index_labels_file = os.path.join("indexes", "csarhiq_s1.txt")
    elif "csarhiqS2" in subset:
        index_labels_file = os.path.join("indexes", "csarhiq_s2.txt")
    elif "csarhiqS3" in subset:
        index_labels_file = os.path.join("indexes", "csarhiq_s3.txt")
    else:
        raise ValueError("Other subsets are not supported")
    with open(index_labels_file, "r") as g:
        pdbs = [line[:4] for line in g.readlines()]
    with open(index_labels_file, "r") as g:
        # file format: pdb code, binding affinity (RT log(Kd/Ki))
        labels = [float(line.split()[1]) for line in g.readlines()]
    protein_files = [os.path.join(data_dir, subset, pdb, "%s_protein.pdb" % pdb) for pdb in pdbs]
    ligand_files = [os.path.join(subset, pdb, "%s_ligand%s" % (pdb, ligfiletype)) for pdb in pdbs]
    
    # Featurize Data
    if para['ifp_para']['ifptype'] in ['splif', 'ecfp', 'plec']:
        feat_t1 = time.time()
        features, failures = featurize_complexes(ligand_files = ligand_files,
                                                 protein_files = protein_files,
                                                 pdbids = pdbs,
                                                 para = para)                                                 
                                     
        feat_t2 = time.time()
        print("\nFeaturization finished, took %0.3f s." % (feat_t2 - feat_t1))
    else:
        raise ValueError("Featurizer not supported")
        return 0
    
    # Delete labels and ids for failing elements
    labels = np.delete(labels, failures)
    labels = labels.reshape((len(labels), 1))
    ids = np.delete(pdbs, failures)

    print("\nConstruct dataset excluding failing featurization elements...")  
    if para['alg_type'] in ['quasi', 'qf_pc']:
        datasets = dc.data.DiskDataset.from_numpy(features, y = labels, ids = ids, data_dir = '/tmp/tmpdt_%s' % subset)
        if train_randsplit_ratio > 0 and train_randsplit_ratio < 1:
            splitter = dc.splits.RandomSplitter()
            train, valid = splitter.train_test_split(datasets, frac_train=train_randsplit_ratio, seed=split_seed)
            all_datasets = (train, valid)
    else:
        for pr in para['folding_para']['power']: 
            datasets[pr] = dc.data.DiskDataset.from_numpy(features[pr], y = labels, ids = ids, data_dir = '/tmp/tmpdt_%s_%s' % (subset, pr))
            if train_randsplit_ratio > 0 and train_randsplit_ratio < 1:
                splitter = dc.splits.RandomSplitter()
                train, valid = splitter.train_test_split(datasets[pr], frac_train=train_randsplit_ratio, seed=split_seed)
                all_datasets[pr] = (train, valid)
                
    if train_randsplit_ratio == 1:
        return (datasets, None)
    elif train_randsplit_ratio > 1 or train_randsplit_ratio <= 0:
        print("Wrong split ratio for training set:\n[0,1] - ratio for trainding split, 0 - cross validation split, 1 - use all for training")
        return (None, None)       
    else:
        return all_datasets


def create_model(tp = 'rf', rand = 0):
    """Initialize a machine-learning model
    Parameters:
        tp - machine-learning approach: 'rf', 'gb', 'nn', 'voting', 'tree' or 'lm'
    Returns a sklearn model
    """
    if tp == 'rf':
        sklearn_model = RandomForestRegressor(random_state = rand, n_estimators = 500)
    elif tp == 'lm':
        sklearn_model = LinearRegression()
    elif tp == 'tree':
        sklearn_model = DecisionTreeRegressor(random_state = rand, max_depth = 10)
    elif tp == 'gb':
        sklearn_model = GradientBoostingRegressor(random_state = rand, n_estimators = 500)
    elif tp == 'nn':
        sklearn_model = MLPRegressor(random_state = rand, max_iter = 500, hidden_layer_sizes = (500,))
    elif tp == 'voting':
        reg1 = GradientBoostingRegressor(random_state = rand, n_estimators = 500)
        reg2 = RandomForestRegressor(random_state = rand, n_estimators = 500)
        reg3 = DecisionTreeRegressor(random_state = rand, max_depth = 10)
        sklearn_model = VotingRegressor(estimators = [('gb', reg1), ('rf', reg2), ('tree', reg3)])
    else:
        print('Wrong model type!!!')
        return []
    
    if tp == 'nn':
        model = dc.models.SklearnModel(sklearn_model, use_weights = False)
    else:
        model = dc.models.SklearnModel(sklearn_model)
    return model


def validate_model(mdl, vset):
    """Evaluate a model on a validation set
    Parameters:
        mdl - a machine-learning model
        vset - a validation set (rows: samples, columns: features)
    Returns the pearson's correlation and RMSE as evaluation results
    """
    test_scores = mdl.evaluate(vset, 
                               metrics = [dc.metrics.Metric(dc.metrics.pearson_r2_score), dc.metrics.Metric(mean_squared_error)], 
                               transformers = [])
    res = {'pc': sqrt(test_scores['pearson_r2_score']), 'rmse': sqrt(test_scores['mean_squared_error'])}
    print("Validation scores: pearsons corr - %.3f, rmse - %.3f.\n" % (res['pc'], res['rmse']))
    return res
