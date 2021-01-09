#%reset -f

import itertools
import numpy as np
from concave_hull import concave_hull_3D
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from deepchem.feat.rdkit_grid_featurizer import hash_ecfp, hash_ecfp_pair


def plec_pairing(plec_degrees):
    """
    Pairs ecfp radii of the two molecular fragments in protein and ligand.
    Parameters:
        plec_degrees - ECFP radii for a pair of molecules
    Returns a list of ECFP-radius pair (e.g. [(0, 0), (1, 1), (2, 1), (3, 1)] if ecfp_protein = 1 and ecfp_ligand = 3)
        note the first index indicates the protein and the second the ligand
    """    
    dg1 = min(plec_degrees)
    dg2 = max(plec_degrees)
    pairings = []
    if plec_degrees[1] == plec_degrees[0]:
        for dg in range(plec_degrees[1] + 1):
            pairings.append((dg, dg))
    else:
        for dg in range(dg1):
            pairings.append((dg, dg))
        pairings += list(itertools.product([dg1], np.arange(dg1, dg2 + 1)) if plec_degrees[0] == dg1
                         else itertools.product(np.arange(dg1, dg2 + 1), [dg1]))   
    pairings = [(int(i), int(j)) for (i,j) in pairings]         
    return pairings


def get_atom_proplist(mol, 
                      sa_dict = None, 
                      aids = [], 
                      base_prop = ['AtomicMass'], 
                      hash_type = 'str'):
    """
    Compute the average properties for a set of atoms in mol (indexed by aids).
    Parameters:
        mol - a rdkit.Chem.rdchem.Mol molecule
        sa_dict - a dictionary mapping atom indices to their solid angles
        aids - the indices of atoms
        base_prop - the property list for the computations
        hash_type - type for hash the properties, can be 'str' or 'vec'
    Returns the computed property list, for 'str' return ['xxx', 'xx.xx', ...] (float number are recorded as %.2f), for 'vec' return the prop list
    """    
    
    tbl = GetPeriodicTable()
    proplist = []
    if len(aids) == 0:
        return proplist
    else:
        proplist = {'AtomicMass': 0, 'TotalConnections': 0, 'HCount': 0, 'HeavyNeighborCount': 0, 'FormalCharge': 0,
                    'DeltaMass': 0, 'SolidAngle': 0, 'SolidAngleValue': 0, 'SolidAngleSign': ''}
        # compute averaged property
        for aid in aids:
            atom = mol.GetAtomWithIdx(aid)
            if 'AtomicMass' in base_prop: 
                proplist['AtomicMass'] += atom.GetMass()
            if 'TotalConnections' in base_prop:
                proplist['TotalConnections'] += atom.GetDegree()
            if 'HCount' in base_prop:
                proplist['HCount'] += atom.GetNumExplicitHs()
            if 'HeavyNeighborCount' in base_prop:
                proplist['HeavyNeighborCount'] += len([bond.GetOtherAtom(atom) for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() > 1])
            if 'FormalCharge' in base_prop:
                proplist['FormalCharge'] += atom.GetFormalCharge()
            if 'DeltaMass' in base_prop:
                proplist['DeltaMass'] += (atom.GetMass() - tbl.GetAtomicWeight(atom.GetAtomicNum()))
            if len([p for p in base_prop if 'SolidAngle' in p]) > 0:
                sa = sa_dict[aid]
                tmp_prop = 0 if (sa is None) else sa
                proplist['SolidAngle'] += tmp_prop
        if 'SolidAngleValue' in base_prop:
            proplist['SolidAngleValue'] = abs(proplist['SolidAngle'])                      
        if 'SolidAngleSign' in base_prop:
            ref = proplist['SolidAngle']
            proplist['SolidAngleSign'] = '0' if ref == 0 else ('+' if ref > 0 else '-')  
        # get str or vec for later hashing
        prop = {k:v for (k, v) in proplist.items() if k in base_prop}
        if hash_type == 'str':
            for key in prop:
                prop[key] = '%.2f' % (prop[key] / len(aids)) if key != 'SolidAngleSign' else prop[key]
        elif hash_type != 'vec':
            print('Wrong hash type!')
            return proplist        
        
        return [v for (k, v) in prop.items()]


def getECFPstringsRadiusN_avg_ecfp(molinfo, 
                                   heavy_atoms = 0,
                                   base_prop = ['AtomicMass'],
                                   sa_dict = {},
                                   indices = None, 
                                   degree = 2,
                                   parameters = {'weighted': 0, 'alpha': -1, 'alpha_step': 0.1},
                                   hash_type = 'str',
                                   idf_power = 64):
    """Obtain molecular fragment for all atoms emanating outward to given degree, using the ECFP procedure.
    For each fragment, compute average atomic properties (and SMILES string for now) and hash to an integer.
    
    Parameters:
        molinfo - a tuple describing a molecule (coordinates, rdkit.Chem.rdchem.Mol molecule, weights), weights = None for non-weighted alpha shapes
        heavy_atoms - use heavy atoms (1) or all atoms (0) to compute ecfp
        base_prop - base atomic property, the hashed value of (base_prop, environment smile) will be the atomic identifiers
               'AtomicMass': the atomic mass of atom
               'TotalConnections': the degree of the atom in the molecule including Hs
               'HeavyNeighborCount': the number of heavy (non-hydrogen) neighbor atoms
               'HCount': the number of attached hydrogens (both implicit and explicit)
               'FormalCharge': the formal charge of atom
               'DeltaMass': the difference between atomic mass and atomic weight (weighted average of atomic masses)
               'SolidAngle': the solid angle of the atom on the molecule surface (> 0: convex, < 0: concave)
               'SolidAngleValue': the absolute value of solid angle of the atom 
               'SolidAngleSign': the sign of solid angle of the atom (-1, 0, 1)
        sa_dict - a dictionary mapping atom indices to their solid angles
        indices - indices for queried atoms
        degree - ecfp radius
        parameters - parameters for calculating the solid angles of surface atoms (for concave_hull_3D class)
        hash_type - type for hashing the fragment, either 'str' (using hash_ecfp function) or 'vec' (using the default hash function)
        idf_power - power for the 'str' hash type (default 64-bit integers)
    Returns a dictionary mapping atom index to a string or vector that is to be hashed later.
  """
    ecfp_dict = {}
    mol = molinfo[1]
    nAtoms = mol.GetNumAtoms()
    neighborhoods = []
    deadAtoms = [0] * nAtoms
    sa_list = sa_dict
    if len([p for p in base_prop if 'SolidAngle' in p]) > 0:
        if len(sa_dict) == 0:
            ch = concave_hull_3D(points = molinfo[0],
                                 weights = molinfo[2],
                                 alpha = parameters['alpha'],
                                 alpha_step = parameters['alpha_step'])
            ch.construct_conchull()
            sa_list = ch.compute_solid_angles()
        else:
            sa_list = sa_dict
        
    aids_all = range(nAtoms) if indices is None else indices  
    for dg in range(degree + 1):
        neighborhoodThisRound = []
        for ix in aids_all:
            i = int(ix)
            if deadAtoms[i] == 0:
                atom = mol.GetAtomWithIdx(i)
                sign1 = (heavy_atoms and atom.GetAtomicNum() == 1)
                sign2 = (atom.GetDegree() == 0)
                if sign1 or sign2:
                    deadAtoms[i] = 1
                    continue
                env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, dg, i, useHs=not heavy_atoms))
                env.sort()
                tmp_aids = set([mol.GetBondWithIdx(bid).GetBeginAtomIdx() for bid in env] +
                                [mol.GetBondWithIdx(bid).GetEndAtomIdx() for bid in env]) 
                env_aids = set([i]) if len(tmp_aids) == 0 else tmp_aids
                tmpprop = get_atom_proplist(mol = mol, sa_dict = sa_list, aids = env_aids, base_prop = base_prop, hash_type = hash_type)
#                submol = Chem.PathToSubmol(mol, env)
#                tmp_smile = Chem.MolToSmiles(submol)
#                smile = atom.GetSymbol() if tmp_smile == '' else tmp_smile
#                tmpprop += [smile]
                # compute idf ############################
                if hash_type == 'str':
                    idf = hash_ecfp(ecfp = ','.join(tmpprop), power = idf_power)
                elif hash_type == 'vec':
                    idf = hash(tuple(tmpprop))
                else:
                    print('Wrong hash type!!!')
                    return ecfp_dict
                ##########################################
                if dg == 0:
                    ecfp_dict[(i, 'r0')] = idf
                else:
                    neighborhoodThisRound.append((env, idf, i))
                    # check if env in the old neighborhood list (previous rounds), if yes turns on the deadAtoms sign
                    if env in neighborhoods:
                        deadAtoms[i] = 1
        if dg > 0:
            neighborhoodThisRound.sort()
            # check if env in the neighborhood list of this round, if yes turns on the deadAtoms sign
            for candidate in neighborhoodThisRound:
                if candidate[0] not in neighborhoods:
                    neighborhoods.append(candidate[0])
                    ecfp_dict[(candidate[2], 'r' + str(dg))] = candidate[1]
                else:
                    deadAtoms[candidate[2]] = 1 # has the same environment as that of atoms in this round
    return ecfp_dict


def getECFPstringsRadiusN_avg_ifp(mols_info, 
                                  heavy_atoms = 0,
                                  base_prop = ['AtomicMass'],
                                  sa_dicts = [{}, {}],
                                  contacts = [[], []], 
                                  ifptype = 'splif',
                                  degrees = [1, 1],
                                  parameters = [{'weighted': 0, 'alpha': -1, 'alpha_step': 0.1},
                                                {'weighted': 0, 'alpha': -1, 'alpha_step': 0.1}],
                                  hash_type = 'str',
                                  idf_power = 64):
    """Obtain pairs of molecular fragments of contacting molecules outward to given degree, using the splif or plec procedure.
    For each fragment in a pair, compute average atomic properties (and SMILES stringfor now) and hash to an integer.
    Parameters:
        mols_info - a list of two tuples each describing a molecule (coordinates, rdkit.Chem.rdchem.Mol molecule, weights)
                    it represents a pair of contacting molecules (e.g. protein and ligand)
        heavy_atoms, base_prop, hash_type and ifp_power same as in getECFPstringsRadiusN_avg_ecfp
        sa_dicts - a list of two dictionaries each mapping atom indices to their solid angles in a molecule
        contacts - a list of two index lists each indicating the queried atoms in a molecule
        ifptype - either 'splif' or 'plec'
        degrees - ecfp radii
        parameters - a list of parameter dictionaries, each for calculating the solid angles of surface atoms of a molecule
    Returns a dictionary mapping atom-pair indices to a string or vector that is to be hashed later.
    """
    ecfp_dict = {}
    mols = [mols_info[0][1], mols_info[1][1]]
    nPairs = len(contacts[0])
    if nPairs == 0:
        print('Wrong contact list!')
        return ecfp_dict
    else:
        neighborhoods = []
        deadAtomPairs = {}
        sa_lists = [{}, {}]
        if len([p for p in base_prop if 'SolidAngle' in p]) > 0:        
            for i in [0, 1]:
                if len(sa_dicts[i]) == 0:
                    tmp = concave_hull_3D(points = mols_info[i][0],
                                          weights = mols_info[i][2],
                                          alpha = parameters[i]['alpha'],
                                          alpha_step = parameters[i]['alpha_step'])
                    tmp.construct_conchull()
                    sa_lists[i] = tmp
                else:
                    sa_lists[i] = sa_dicts[i]
            
        if ifptype == 'splif':
            dg_pairs = [(degrees[0], degrees[1])]
        elif ifptype == 'plec':            
            dg_pairs = plec_pairing(plec_degrees = degrees)
        else:
            print('Wrong ifp type!')
            return ecfp_dict
        
        for dgs in dg_pairs:
            neighborhoodThisRound = []
            for (a1, a2) in zip(contacts[0], contacts[1]):
                inds = (int(a1), int(a2))
                if inds not in deadAtomPairs:
                    atoms = (mols[0].GetAtomWithIdx(inds[0]), mols[1].GetAtomWithIdx(inds[1]))
                    sign1 = (heavy_atoms and (atoms[0].GetAtomicNum() == 1 or atoms[1].GetAtomicNum() == 1))
                    sign2 = (atoms[0].GetDegree() == 0 or atoms[1].GetDegree() == 0)
                    if sign1 or sign2:
                        deadAtomPairs[inds] = 1
                        continue
                    nbhd_pairs = []
                    for k in [0, 1]:
                        env = list(Chem.FindAtomEnvironmentOfRadiusN(mols[k], dgs[k], inds[k], useHs=not heavy_atoms))
                        env.sort()
                        tmp_aids = set([mols[k].GetBondWithIdx(bid).GetBeginAtomIdx() for bid in env] +
                                        [mols[k].GetBondWithIdx(bid).GetEndAtomIdx() for bid in env])
                        env_aids = set([inds[k]]) if len(tmp_aids) == 0 else tmp_aids
                        tmpprop = get_atom_proplist(mol = mols[k], sa_dict = sa_lists[k], aids = env_aids, base_prop = base_prop, hash_type = hash_type)
#                        submol = Chem.PathToSubmol(mols[k], env)
#                        tmp_smile = Chem.MolToSmiles(submol)
#                        smile = atoms[k].GetSymbol() if tmp_smile == '' else tmp_smile
#                        tmpprop += [smile]
                        nbhd_pairs.append((env, tmpprop, inds[k]))
                    if dgs == (0, 0):
                        if hash_type == 'str':
                            tobehashed = (','.join(nbhd_pairs[0][1]), ','.join(nbhd_pairs[1][1]))
                            idf = hash_ecfp_pair(ecfp_pair = tobehashed, power = idf_power)
                        elif hash_type == 'vec':
                            tobehashed = (tuple(nbhd_pairs[0][1]), tuple(nbhd_pairs[1][1]))
                            idf = hash(tobehashed)
                        else:
                            print('Wrong hash type!!!')
                            return ecfp_dict
                        ecfp_dict[(inds, 'r0-r0')] = idf
                    else:
                        neighborhoodThisRound.append(nbhd_pairs)
                        if (nbhd_pairs[0][0], nbhd_pairs[1][0]) in neighborhoods:
                            deadAtomPairs[inds] = 1
            if dgs != (0, 0):
                neighborhoodThisRound.sort()
                for candidate in neighborhoodThisRound:
                    envs = (candidate[0][0], candidate[1][0])
                    cand_inds = (candidate[0][2], candidate[1][2])
                    if envs not in neighborhoods:
                        neighborhoods.append(envs)
                        if hash_type == 'str':
                            tobehashed = (','.join(candidate[0][1]), ','.join(candidate[1][1]))
                            idf = hash_ecfp_pair(ecfp_pair = tobehashed, power = idf_power)
                        elif hash_type == 'vec':
                            tobehashed = (tuple(candidate[0][1]), tuple(candidate[1][1]))
                            idf = hash(tobehashed)
                        else:
                            print('Wrong hash type!!!')
                            return ecfp_dict
                        ecfp_dict[(cand_inds, 'r' + str(dgs[0]) + '-r' + str(dgs[1]))] = idf
                    else:
                        deadAtomPairs[cand_inds] = 1 
        return ecfp_dict

                        
def getECFPidentifiers_molpair_avg(mols, 
                                   heavy_atoms = 0,
                                   base_prop = ['AtomicMass'],
                                   sa_dicts = [{}, {}],
                                   contacts = [[], []], 
                                   ifptype = 'ecfp',
                                   degrees = [2, 2],
                                   parameters = [{'weighted': 0, 'alpha': -1, 'alpha_step': 0.1},
                                                 {'weighted': 0, 'alpha': -1, 'alpha_step': 0.1}],
                                   hash_type = 'str',
                                   idf_power = 64):
    """Obtain the integer identifers of molecular fragments.
    """
    idf_list = [[], []]
    idfs = []
    if ifptype in ['ecfp']:
        for i in [0, 1]:
            tmp = getECFPstringsRadiusN_avg_ecfp(molinfo = mols[i], 
                                                 heavy_atoms = heavy_atoms,
                                                 base_prop = base_prop,
                                                 sa_dict = sa_dicts[i],
                                                 indices = contacts[i],
                                                 degree = degrees[i],
                                                 parameters = parameters[i],
                                                 hash_type = hash_type,
                                                 idf_power = idf_power)
            idf_list[i] = list(tmp.values())
        return idf_list
    elif ifptype in ['splif', 'plec']:
        tmp = getECFPstringsRadiusN_avg_ifp(mols_info = mols, 
                                    heavy_atoms = heavy_atoms,
                                    base_prop = base_prop,
                                    sa_dicts = sa_dicts,
                                    contacts = contacts, 
                                    ifptype = ifptype,
                                    degrees = degrees,
                                    parameters = parameters,
                                    hash_type = hash_type,
                                    idf_power = idf_power)
        idfs = list(tmp.values())
        return idfs
    else:
        print('Wrong ifptype!')
        return []


