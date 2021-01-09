#%reset -f

import itertools
import numpy as np
#from concave_hull import concave_hull_3D
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from deepchem.feat.rdkit_grid_featurizer import hash_ecfp


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


def getOriginalIdentifiers(mol, 
                           prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                   'DeltaMass', 'IsTerminalAtom',
                                   'SolidAngle', 'SolidAngleValue', 'SolidAngleSign'],
                           sa_dict = None,
                           includeAtoms = None,
                           radius = 2,
                           hash_type = 'str',
                           idf_power = 64):
    """Compute the original identifiers for atoms in a molecule based on atomic properties. 
       Note it only includes HEAVY atoms.
    Parameters:
        mol - rdkit.Chem.rdchem.Mol molecule
        prop - atomic property list
               'AtomicNumber': the atomic number of atom
               'AtomicMass': the mass of atom
               'TotalConnections': the degree of the atom in the molecule including Hs
               'HeavyNeighborCount': the number of heavy (non-hydrogen) neighbor atoms
               'HCount': the number of attached hydrogens (both implicit and explicit)
               'FormalCharge': the formal charge of atom
               'DeltaMass': the difference between atomic mass and atomic weight (weighted average of atomic masses)
               'IsTerminalAtom': indicates whether the atom is a terminal atom
               'SolidAngle': the solid angle of the atom on the molecule surface (> 0: convex, < 0: concave)
               'SolidAngleValue': the absolute solid angle of the atom on the molecule surface
               'SolidAngleSign': the sign of solid angle of the atom (-1, 0, 1)
        sa_dict - a dictionary mapping atom indices to their solid angles
        includeAtoms - atom indices for getting identifiers
        radius - ECFP radius, only calculates the identifiers of atoms in the neighborhoods (of radius) of included atoms (includeAtoms)
        hash_type - type for hash the properties, can be 'str' or 'vec'
        idf_power - power for the 'str' hash type (default 64-bit integers)
    Returns an dictionary mapping each heavy-atom index to an integer representing the atomic properties
    """
    tbl = GetPeriodicTable()
    idf_dict = {}
    nAtoms = mol.GetNumAtoms()
    if includeAtoms is None:
        indices = range(nAtoms)
    else:
        indices = includeAtoms
    for i in indices:
        index = int(i)
        env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, index, useHs=True))
        env_aids = set([mol.GetBondWithIdx(bid).GetBeginAtomIdx() for bid in env] + [mol.GetBondWithIdx(bid).GetEndAtomIdx() for bid in env]) 
        for aid in env_aids:
            if (aid, 0) not in idf_dict:
                atom = mol.GetAtomWithIdx(aid)        
                if atom.GetAtomicNum() > 1:
                    properties = []
                    if 'AtomicNumber' in prop:
                        properties.append(atom.GetAtomicNum())
                    if 'AtomicMass' in prop:
                        tmp_prop = atom.GetMass() if hash_type == 'vec' else '%.2f' % atom.GetMass()
                        properties.append(tmp_prop)
                    if 'TotalConnections' in prop:
                        properties.append(atom.GetDegree())
                    if 'HCount' in prop:
                        properties.append(atom.GetNumExplicitHs())
                    if 'HeavyNeighborCount' in prop:
                        properties.append(len([bond.GetOtherAtom(atom) for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() > 1]))
                    if 'FormalCharge' in prop:
                        tmp_prop = atom.GetFormalCharge() if hash_type == 'vec' else '%.2f' % atom.GetFormalCharge()
                        properties.append(tmp_prop)
                    if 'DeltaMass' in prop:
                        tmp_prop = atom.GetMass() - tbl.GetAtomicWeight(atom.GetAtomicNum())
                        tmp_prop = tmp_prop if hash_type == 'vec' else '%.2f' % tmp_prop
                        properties.append()
                    if 'IsTerminalAtom' in prop:
                        is_terminal_atom = 1 if atom.GetDegree() == 1 else 0
                        properties.append(is_terminal_atom)
                    if len([p for p in prop if 'SolidAngle' in p]) > 0:
                        sa = sa_dict[aid]
                        solang = 0 if (sa is None) else sa
                        if 'SolidAngle' in prop:
                            tmp_prop = solang if hash_type == 'vec' else '%.2f' % solang
                            properties.append(tmp_prop)
                        elif 'SolidAngleValue' in prop:
                            tmp_prop = abs(solang) if hash_type == 'vec' else '%.2f' % abs(solang)
                            properties.append(tmp_prop)
                        else:
                            solang_sign = '0' if (sa in [None, 0]) else ('+' if sa > 0 else '-')
                            properties.append(solang_sign)
                    
                    if hash_type == 'str':
                        idf = hash_ecfp(ecfp = ','.join([str(p) for p in properties]), power = idf_power)
                    elif hash_type == 'vec':
                        idf = hash(tuple(properties))
                    else:
                        print('Wrong hash type!')
                        return idf_dict
                        
                    idf_dict[(aid, 0)] = idf

    return idf_dict

def getIdentifiersRadiusN(molinfo, 
                          prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                  'DeltaMass', 'IsTerminalAtom',
                                  'SolidAngle', 'SolidAngleValue', 'SolidAngleSign'],
                          sa_dict = None,
                          includeAtoms = None,
                          radius = 2,
                          hash_type = 'str',
                          idf_power = 64):
    """Calculate the Identifiers of molecular fragments (each originated from an atom, of radius N) in a molecule.
    Parameters:
        molinfo - a tuple describing a molecule (coordinates, rdkit.Chem.rdchem.Mol molecule, weights), weights = None for non-weighted alpha shapes
        prop, sa_dict, radius, includeAtoms, hash_type and idf_power - same as in getOriginalIdentifiers
    Returns the identifiers
    """
    res_idfs = []
    mol = molinfo[1]
    nAtoms = mol.GetNumAtoms()
    neighborhoods = []
    deadAtoms = [0] * nAtoms
    
    # get original identifiers (of radius 0) of included atoms and their neighbors (in neighborhood of radius)
    idf_dict = getOriginalIdentifiers(mol = mol, 
                                      prop = prop,
                                      sa_dict = sa_dict,
                                      includeAtoms = includeAtoms,
                                      radius = radius,
                                      hash_type = hash_type,
                                      idf_power = idf_power)
    ids_fil = set([u[0] for (u,v) in idf_dict.items()])

    # get atom orders
    if includeAtoms is not None:
        idfs = {u:v for (u,v) in idf_dict.items() if u[0] in includeAtoms}
        # put the query atoms in front positions (access first)
        atomOrder = includeAtoms + [i for i in ids_fil if i not in includeAtoms]
    else:
        idfs = idf_dict
        atomOrder = range(nAtoms)
    # initialize res_idfs
    res_idfs += list(idfs.values())
    
    # iteratively calculate the identifiers of larger radius
    if radius == 0:
        return res_idfs
    else:
        for layer in range(radius):
            round_idfs = {}
            neighborhoodThisRound = []
            for ind in atomOrder:
                index = int(ind)
                if not deadAtoms[index]:
                    atom = mol.GetAtomWithIdx(index)
                    env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, layer + 1, index, useHs=True))
                    env.sort()
                    if atom.GetAtomicNum() == 1 or atom.GetDegree == 0:
                        deadAtoms[index] = 1
                        continue
                    nbrs = []
                    bonds = atom.GetBonds()
                    for bond in bonds:
                        oth_index = bond.GetOtherAtomIdx(index)
                        if (oth_index, layer) in idf_dict:
                            bt = bond.GetBondTypeAsDouble()
                            nbrs.append((bt, idf_dict[(oth_index, layer)]))
                    nbrs.sort()
                    nbrhd = [layer, idf_dict[(index, layer)]]
                    for nbr in nbrs:                        
                        nbrhd.append(nbr)
                    # use [layer, idf, (nbr1_bondtype, nbr1_idf), ..., (nbrN_bondtype, nbrN_idf)] to represent an atomic neighborhood of a specific radius (layer)
                    if hash_type == 'str':
                        idf = hash_ecfp(ecfp = ','.join([str(itm) for itm in nbrhd]), power = idf_power)
                    elif hash_type == 'vec':
                        idf = hash(tuple(nbrhd))
                    else:
                        print('Wrong hash type!!!')
                        return []
                    
                    round_idfs[(index, layer + 1)] = idf
                    neighborhoodThisRound.append((env, idf, index))
                    if env in neighborhoods:
                        deadAtoms[index] = 1
            
            neighborhoodThisRound.sort()
            for candi in neighborhoodThisRound:
                if candi[0] not in neighborhoods:
                    neighborhoods.append(candi[0])
                    res_idfs.append(candi[1])
                else:
                    deadAtoms[candi[2]] = 1
            idf_dict = round_idfs
        
        return res_idfs
    

def getIdentifiersRadiusN_all(molinfo, 
                              prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                      'DeltaMass', 'IsTerminalAtom',
                                      'SolidAngle', 'SolidAngleValue', 'SolidAngleSign'],
                              sa_dict = None,
                              includeAtoms = None,
                              radius = 2,
                              hash_type = 'str',
                              idf_power = 64):
    """Calculate the Identifiers of all molecular fragments (each originated from an atom, of radius N, can be redundant) in a molecule.
    Parameters:
        molinfo - a tuple describing a molecule (coordinates, rdkit.Chem.rdchem.Mol molecule, weights), weights = None for non-weighted alpha shapes
        prop, sa_dict, radius, includeAtoms, hash_type and idf_power - same as in getOriginalIdentifiers
    Returns the identifiers
    """
    idfs_all = {}
    mol = molinfo[1]
    nAtoms = mol.GetNumAtoms()
    deadAtoms = [0] * nAtoms
    
    # get original identifiers (of radius 0) of included atoms and their neighbors (in neighborhood of radius)
    idf_dict = getOriginalIdentifiers(mol = mol, 
                                      prop = prop,
                                      sa_dict = sa_dict,
                                      includeAtoms = includeAtoms,
                                      radius = radius,
                                      hash_type = hash_type,
                                      idf_power = idf_power)
    ids_fil = set([u[0] for (u,v) in idf_dict.items()])
    idfs_all = {k: (v, []) for (k, v) in idf_dict.items()}

    # get atom orders
    if includeAtoms is not None:
        # put the query atoms in front positions (access first)
        atomOrder = includeAtoms + [i for i in ids_fil if i not in includeAtoms]
    else:
        atomOrder = range(nAtoms)
    
    # iteratively calculate the identifiers of larger radius
    if radius == 0:
        return idfs_all
    else:
        for layer in range(radius):
            for ind in atomOrder:
                index = int(ind)
                if not deadAtoms[index]:
                    atom = mol.GetAtomWithIdx(index)
                    env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, layer + 1, index, useHs=True))
                    env.sort()
                    if atom.GetAtomicNum() == 1 or atom.GetDegree == 0:
                        deadAtoms[index] = 1
                        continue
                    nbrs = []
                    bonds = atom.GetBonds()
                    for bond in bonds:
                        oth_index = bond.GetOtherAtomIdx(index)
                        if (oth_index, layer) in idfs_all:
                            bt = bond.GetBondTypeAsDouble()
                            nbrs.append((bt, idfs_all[(oth_index, layer)][0]))
                    nbrs.sort()
                    nbrhd = [layer, idfs_all[(index, layer)][0]]
                    for nbr in nbrs:                        
                        nbrhd.append(nbr)
                    # use [layer, idf, (nbr1_bondtype, nbr1_idf), ..., (nbrN_bondtype, nbrN_idf)] to represent an atomic neighborhood of a specific radius (layer)
                    if hash_type == 'str':
                        idf = hash_ecfp(ecfp = ','.join([str(itm) for itm in nbrhd]), power = idf_power)
                    elif hash_type == 'vec':
                        idf = hash(tuple(nbrhd))
                    else:
                        print('Wrong hash type!!!')
                        return []
                    
                    idfs_all[(index, layer + 1)] = (idf, env)
        
        return idfs_all



def getIdentifiersRadiusN_ifp(mols_info, 
                              prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                      'DeltaMass', 'IsTerminalAtom',
                                      'SolidAngle', 'SolidAngleValue', 'SolidAngleSign'],
                              sa_dicts = [{}, {}],
                              contacts = [[], []],
                              degrees = [1, 1],
                              hash_type = 'str',
                              idf_power = 64,
                              ifptype = 'splif'):
    """Computes SPLIF identifiers for a pair of molecular fragments (e.g. protein-binding pocket and ligand).
    Parameters:
        mols_info - a list of two molecules (coordinates, rdkit.Chem.rdchem.Mol molecule, weights)
        sa_dicts - a list of dictionaries each mapping atom indices to their solid angles (e.g. sa_dicts[0] for protein, sa_dicts[1] for ligand)
        contacts - a list of two sets, each indicating the indices of atoms to be considered in a molecule
        degrees - ecfp radii
        prop, hash_type and idf_power - same as above
        ifptype - either 'splif' or 'plec'
    """
    idf_dicts = [{}, {}]
    res_idfs = []
    mols = [mols_info[0][1], mols_info[1][1]]
    nPairs = len(contacts[0])
    if nPairs == 0:
        print('Wrong contact list!')
        return res_idfs
    else:
        neighborhoods = []
        deadAtomPairs = {}        
            
        if ifptype == 'splif':
            dg_pairs = [(degrees[0], degrees[1])]
        elif ifptype == 'plec':            
            dg_pairs = plec_pairing(plec_degrees = degrees)
        else:
            print('Wrong ifp type!')
            return res_idfs
        
        # get original identifiers of included atoms
        for k in [0, 1]:
            idf_dicts[k] = getIdentifiersRadiusN_all(molinfo = mols_info[k],
                     prop = prop,
                     sa_dict = sa_dicts[k],
                     includeAtoms = contacts[k],
                     radius = degrees[k],
                     hash_type = hash_type,
                     idf_power = idf_power)           
        
        for dgs in dg_pairs:
            for (a1, a2) in zip(contacts[0], contacts[1]):
                inds = (int(a1), int(a2))
                if inds not in deadAtomPairs:
                    atoms = (mols[0].GetAtomWithIdx(inds[0]), mols[1].GetAtomWithIdx(inds[1]))
                    sign1 = (atoms[0].GetAtomicNum() == 1 or atoms[1].GetAtomicNum() == 1)
                    sign2 = (atoms[0].GetDegree() == 0 or atoms[1].GetDegree() == 0)
                    if sign1 or sign2:
                        deadAtomPairs[inds] = 1
                        continue
                    envs = [0, 0]
                    nhds = []
                    for m in [0, 1]:
                        nhds.append(idf_dicts[m][(inds[m], dgs[m])])
                        if dgs[m] == 0:
                            envs[m] = str(inds[m])
                        else:
                            envs[m] = nhds[m][1]
                    if envs not in neighborhoods:
                        neighborhoods.append(envs)
                        res_idfs.append(hash(tuple((nhds[0][0], nhds[1][0]))))
            
            return res_idfs


def getECFPidentifiers_molpair_classic(mols_info, 
                                       prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                               'DeltaMass', 'IsTerminalAtom',
                                               'SolidAngle', 'SolidAngleValue', 'SolidAngleSign'],
                                       sa_dicts = [{}, {}],
                                       contacts = [[], []],
                                       degrees = [1, 1],
                                       hash_type = 'str',
                                       idf_power = 64,
                                       ifptype = 'splif'):
    """Obtain the integer identifers of molecular fragments.
    """
    idf_list = [[], []]
    idfs = []
    if ifptype in ['ecfp']:
        for i in [0, 1]:
            idf_list[i] = getIdentifiersRadiusN(molinfo = mols_info[i],
                    prop = prop,
                    sa_dict = sa_dicts[i],
                    includeAtoms = contacts[i],
                    radius = degrees[i],
                    hash_type = hash_type,
                    idf_power = idf_power)
        return idf_list
    elif ifptype in ['splif', 'plec']:
        idfs = getIdentifiersRadiusN_ifp(mols_info = mols_info, 
                                         prop = prop,
                                         sa_dicts = sa_dicts,
                                         contacts = contacts,
                                         degrees = degrees,
                                         hash_type = hash_type,
                                         idf_power = idf_power,
                                         ifptype = ifptype)
        return idfs
    else:
        print('Wrong ifptype!')
        return []
