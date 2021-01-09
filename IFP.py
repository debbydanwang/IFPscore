#%reset -f

import os
import numpy as np
from concave_hull import concave_hull_3D
from deepchem.utils.rdkit_util import load_molecule
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
import itertools
import warnings
from rdkit.Chem import GetPeriodicTable
from Bio.PDB import Select
from PDB_rdkit_util_avg import getECFPidentifiers_molpair_avg
from PDB_rdkit_util import getECFPidentifiers_molpair_classic
from pickle5 import dump, load
    
def get_point_weight(mol, probe_rad = 1.4):
    """Form a set of weighted points, with each point representing an atom in a molecule.
    Parameters:
        mol - rdkit.Chem.rdchem.Mol molecule
        probe_rad - probe radius, default of 1.4 Angstrom
    Returns the weights of the points (squared (VDW radius + probe radius)).
    """
    coor = mol.GetConformer().GetPositions()
    tbl = GetPeriodicTable()
    pts_num = coor.shape[0]
    weights = np.zeros(pts_num)
    for i in range(pts_num):
        weights[i] = (tbl.GetRvdw(mol.GetAtomWithIdx(int(i)).GetAtomicNum()) + probe_rad) ** 2
    return weights

def select_protein_chains(pro_bio, lig_xyz, dist_cutoff = 4.5):
    """Selects the chains and hetero atoms close to them.
    Parameters:
        pro_bio - a structure parsed by Bio.PDB.PDBParser
        lig_xyz - coordinates for the ligand, size of (n, 3) with n indicating the atoms in the ligand
        dist_cutoff - if the average distance between a chain and the ligand is in range [min, min + dist_cutoff] then select this chain,
                      where min is the minimum distance between a chain and the ligand
        Returns a list of chain ids and a dictionary of the hetatms close to each selected chain (with the residue number and atom id)
    """
    # select protein chains according to their average atomic distances with the ligand
    chids_full = [i.get_id() for i in pro_bio[0].get_list()]
    chids = [i for i in chids_full if i != ' ']
    chain_coords = {chain:[] for chain in chids}
    chain_dist = {chain:0 for chain in chids}
    for i in chain_coords:
        for atom in pro_bio[0][i].get_atoms():
            chain_coords[i].append(atom.get_coord())
        tmp_coords = np.vstack(chain_coords[i])
        chain_dist[i] = np.mean(cdist(tmp_coords, lig_xyz, metric = 'euclidean'))
    dist_min = min(list(chain_dist.values()))
    selected = []
    for i in chain_dist:
        if chain_dist[i] - dist_min < dist_cutoff:
            selected.append(i)
    # select hetero atoms according to type and distance
    hetatm_for_chains = {chain:[] for chain in chids}
    if ' ' in chids_full:
        for hetatm in pro_bio[0][' '].get_atoms():
            if hetatm.get_parent().get_resname() != 'HOH':
                avg_dist = {chain:0 for chain in chids}
                for i in chain_coords:
                    tmp_coords = np.vstack(chain_coords[i])
                    avg_dist[i] = np.mean(cdist(tmp_coords, np.reshape(hetatm.get_coord(), (1, -1)), metric = 'euclidean'))
                hetatm_for_chains[min(avg_dist, key = avg_dist.get)].append((hetatm.get_parent().get_id()[1], hetatm.get_id()))
    
    return {chain:hetatm_for_chains[chain] for chain in selected}

class PartnerSelect(Select):
    """It is a superclass of the Bio.PDB.Select class.
    Parameters for __init__ function:
        chain_dict - a dictionary with the selected chain id as the key and the close hetatms as the value (with residue number and atom id info)
                     e.g. {'A': [(1, 'CA'), (2, 'CA')]}
    """    
    def __init__(self, chain_dict = {}):
        self.selected_chain = list(chain_dict.keys())[0]
        self.hetatm = list(chain_dict.values())[0]

    def accept_residue(self, residue):
        chid = residue.get_parent().get_id()
        resid = residue.get_id()[1]
        selected_hetatm_ids = [i for (i, j) in self.hetatm]
        if chid == self.selected_chain:
            return True
        elif chid == ' ' and resid in selected_hetatm_ids:
            return True
        else:
            return False


def ifp_folding(identifiers = [],
                channel_power = 10,
                counts = 0):
    """
    Folds a list of integer identifiers to a bit vector of fixed-length.
    Parameters:
        identifiers - a list of integers
        channel_power - decides the length of feature vector
        counts - use occurences of identifiers (1) or not (0)
    Returns a final feature vector of length 2^channel_power
    """
    feature_vector = np.zeros(2**channel_power)
    on_channels = [int(idf % 2**channel_power) for idf in identifiers]
    if counts:
        for ch in on_channels:
            feature_vector[ch] += 1
    else:
        feature_vector[on_channels] += 1

    return feature_vector


class pro_lig_ifp(object):
    def __init__(self, 
                 fn_pro, 
                 fn_lig, 
                 pdb = None,
                 int_cutoff = 4.5,
                 ch_para = {'weighted': 0, 'alpha': [-1, -1], 'alpha_step': [0.1, 0.1]},
                 contact_bins = [(1, 2), (2, 3), (3, 4.5)]):
        """
        Initialize a protein-ligand interaction class.
        Parameters:
            fn_pro - the file name of a protein for loading.
            fn_lig - the file name of a ligand for loading.
            pdb - pdb id of the molecule under processing
            int_cutoff - distance threshold for locating interaction atoms in protein and ligand
            alpha - alpha value for contructing concave hulls of protein and ligand 
                    (if alpha = -1, this parameter will be tuned according to alpha_step)
            alpha_step - steps for tuning alpha value in the construction of concave hulls
            contact_bins - distance bins for identifying different atom contacts at protein-ligand interfaces
        """
        self.pdb = pdb if pdb is not None else "comp"
        print('Constructing an ifp object for PDB:%s.........' % self.pdb)
        parser = PDBParser(PERMISSIVE = 1)
        warnings.filterwarnings("ignore")
        pro_bio = parser.get_structure(pdb, fn_pro)
        self.weighted = ch_para['weighted']
        # reading the ligand ##############################################################
        self.lig = (load_molecule(fn_lig, add_hydrogens=False, calc_charges=False, sanitize=False)) # PDBbind data already include Hs, but need to be sanitized for calculating ring memberships for atoms
        if self.weighted == 1:
            self.lig = (self.lig[0], self.lig[1], get_point_weight(mol = self.lig[1]))
        else:
            self.lig = (self.lig[0], self.lig[1], None)
        # reading each chain and save the cleaned pdb file ###############################
        self.chain_dict = select_protein_chains(pro_bio = pro_bio, lig_xyz = self.lig[0], dist_cutoff = int_cutoff)
        self.pro = {chain: None for chain in self.chain_dict}
        for chain in self.chain_dict:
            fn_chain = fn_pro.replace('_protein.pdb', '_protein_chain%s.pdb' % chain)
            if not os.path.isfile(fn_chain):
                print('Saving chain %s...' % chain)
                tmp_dict = {k:v for (k, v) in self.chain_dict.items() if k == chain}
                io = PDBIO()
                io.set_structure(pro_bio)
                io.save(fn_chain, PartnerSelect(tmp_dict))
            self.pro[chain] = (load_molecule(fn_chain, add_hydrogens=False, calc_charges=False, sanitize=False))
            if self.weighted == 1:
                self.pro[chain] = (self.pro[chain][0], self.pro[chain][1], get_point_weight(mol = self.pro[chain][1]))
            else:
                self.pro[chain] = (self.pro[chain][0], self.pro[chain][1], None)
        ###################################################################################
        self.cutoff = int_cutoff
        self.alpha = ch_para['alpha']
        self.alpha_step = ch_para['alpha_step']
        self.contact_bins = contact_bins
        self.pairwise_distances = {chain: None for chain in self.chain_dict}
        self.contacts = {chain: None for chain in self.chain_dict}
        # loading or calculating the solid angle lists ###################################
        self.sadicts = {}
        substr = fn_lig[fn_lig.find('_ligand.'):len(fn_lig)]
        fn_lig_sa = fn_lig.replace(substr, '_ligand_ch_wt%s.pkl' % self.weighted)
        if not os.path.isfile(fn_lig_sa):
            tmp = concave_hull_3D(points = self.lig[0], weights = self.lig[2], alpha = self.alpha[1], alpha_step = self.alpha_step[1])
            tmp.construct_conchull()
            sadict_lig = tmp.compute_solid_angles()
            a_file = open(fn_lig_sa, "wb")
            dump(sadict_lig, a_file)
            a_file.close()
        else:
            a_file = open(fn_lig_sa, "rb")
            sadict_lig = load(a_file)
            a_file.close()
        for chain in self.chain_dict:
            fn_chain_sa = fn_pro.replace('_protein.pdb', '_protein_chain%s_ch_wt%s.pkl' % (chain, self.weighted))
            if not os.path.isfile(fn_chain_sa):
                tmp = concave_hull_3D(points = self.pro[chain][0], weights = self.pro[chain][2], alpha = self.alpha[0], alpha_step = self.alpha_step[0])
                tmp.construct_conchull()
                sadict_pro = tmp.compute_solid_angles()
                a_file = open(fn_chain_sa, "wb")
                dump(sadict_pro, a_file)
                a_file.close()
            else:
                a_file = open(fn_chain_sa, "rb")
                sadict_pro = load(a_file)
                a_file.close()
            self.sadicts[chain] = [sadict_pro, sadict_lig]

     
    def find_contacts(self):
        """
        Use the 3D coords of protein chains (self.pro) and ligand (self.lig)
        to calculate the pairwise distances (m * n arrary) between the atoms,
        and find the interacting atoms based on self.cutoff (in Angstrom),
        this function updates self.contacts that includes the atom indices of 
        those interacting atoms in protein and ligand 
        """
        for chain in self.pro:
            self.pairwise_distances[chain] = cdist(self.pro[chain][0], self.lig[0], metric = 'euclidean')
            self.contacts[chain] = np.nonzero(self.pairwise_distances[chain] < self.cutoff)
    
    def get_quasi_fragmental_desp(self, bins = None):
        """
        Compute quasi fragmental descriptors.
        """
        if bins is not None:
            self.contact_bins = bins
        pool = [5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 23, 25, 26, 27, 28, 29, 30, 33, 34, 35, 48, 53, 70, 79, 80, 82, 92]
        quasi_type = list(itertools.product(pool, pool))
        quasi_feat = []
        for cbin in self.contact_bins:
            occur = {}
            for tp in quasi_type:
                occur[tp] = 0
            for chain in self.chain_dict:
                contacts = np.nonzero((self.pairwise_distances[chain] > cbin[0]) & (self.pairwise_distances[chain] < cbin[1]))
                conts = zip(contacts[0], contacts[1])
                for cont in conts:
                    atm1 = self.pro[chain][1].GetAtomWithIdx(int(cont[0]))
                    atm2 = self.lig[1].GetAtomWithIdx(int(cont[1]))
                    tmp = (atm1.GetAtomicNum(), atm2.GetAtomicNum())
                    if tmp in quasi_type:
                        occur[tmp] += 1
            
            quasi_feat += list(occur.values())    
        
        return quasi_feat      
    
    def get_qf_pc_desp(self, bins = None):
        """
        Compute quasi fragmental descriptors.
        """
        if bins is not None:
            self.contact_bins = bins
        quasi_type = [5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 23, 25, 26, 27, 28, 29, 30, 33, 34, 35, 48, 53, 70, 79, 80, 82, 92]
        quasi_feat = []
        for cbin in self.contact_bins:
            occur_pro = {}
            occur_lig = {}
            for tp in quasi_type:
                occur_pro[tp] = 0
                occur_lig[tp] = 0
            for chain in self.chain_dict:
                contacts = np.nonzero((self.pairwise_distances[chain] > cbin[0]) & (self.pairwise_distances[chain] < cbin[1]))
                conts = zip(contacts[0], contacts[1])
                for cont in conts:
                    atm1 = self.pro[chain][1].GetAtomWithIdx(int(cont[0]))
                    atm2 = self.lig[1].GetAtomWithIdx(int(cont[1]))
                    tmp1 = atm1.GetAtomicNum()
                    tmp2 = atm2.GetAtomicNum()
                    if tmp1 in quasi_type:
                        occur_pro[tmp1] += 1
                    if tmp2 in quasi_type:
                        occur_lig[tmp2] += 1
            
            quasi_feat += (list(occur_pro.values()) + list(occur_lig.values()))
        
        return quasi_feat      

    def plot_concave_hull(self, 
                          indices = None,
                          interaction = 0,
                          comb = 1):
        """
        Plot the concave hulls for the protein and ligand 
        """
        fig = plt.figure()
        idx = 0
        for chain in self.pro:
            idx += 1
            ch_pro = concave_hull_3D(name = self.pdb + '_pro_chain%s' % chain, 
                                     points = self.pro[chain][0], 
                                     weights = self.pro[chain][2],
                                     alpha = self.alpha[0], 
                                     alpha_step = self.alpha_step[0])
            ch_pro.construct_conchull()
            ch_lig = concave_hull_3D(name = self.pdb + '_lig', 
                                     points = self.lig[0], 
                                     weights = self.lig[2],
                                     alpha = self.alpha[1], 
                                     alpha_step = self.alpha_step[1])
            ch_lig.construct_conchull()
            
            if interaction == 1:
                ind_pro = np.concatenate([np.where(ch_pro.Triangles == i)[0] for i in self.contacts[chain][0]])
                ind_lig = np.concatenate([np.where(ch_lig.Triangles == j)[0] for j in self.contacts[chain][1]])
                tri_pro = ch_pro.Triangles[np.unique(ind_pro)]
                tri_lig = ch_lig.Triangles[np.unique(ind_lig)]   
            else:
                tri_pro = ch_pro.Triangles
                tri_lig = ch_lig.Triangles
            
            if comb == 1:
                ax = fig.add_subplot(1, len(list(self.pro.keys())), idx, projection = '3d')
                ax.plot_trisurf(self.pro[chain][0][:,0], self.pro[chain][0][:,1], self.pro[chain][0][:,2],
                                triangles = tri_pro, cmap = plt.cm.Blues, edgecolor = 'b')
                ax.plot_trisurf(self.lig[0][:,0], self.lig[0][:,1], self.lig[0][:,2], 
                                triangles = tri_lig, cmap = plt.cm.Oranges, edgecolor = 'r')
            else:
                ax = fig.add_subplot(len(list(self.pro.keys())), 2, idx * 2 - 1, projection = '3d')
                ax.plot_trisurf(self.pro[chain][0][:,0], self.pro[chain][0][:,1], self.pro[chain][0][:,2],
                                triangles = tri_pro, cmap = plt.cm.Blues, edgecolor = 'b')
                ax = fig.add_subplot(len(list(self.pro.keys())), 2, idx * 2, projection = '3d')
                ax.plot_trisurf(self.lig[0][:,0], self.lig[0][:,1], self.lig[0][:,2], 
                                triangles = tri_lig, cmap = plt.cm.Oranges, edgecolor = 'r')
            
        plt.show()
        
    
    def compute_ifp_features_in_range(self, 
                                      contact_bin, 
                                      ifptype = 'ecfp', 
                                      alg_type = 'avg',
                                      ch_para = [{'weighted': 0, 'alpha': -1, 'alpha_step': 0.1},
                                                 {'weighted': 0, 'alpha': -1, 'alpha_step': 0.1}],
                                      ecfp_radius = [2, 2],
                                      heavy_atoms = 0,
                                      base_prop = ['AtomicMass'],
                                      hash_type = 'str',
                                      idf_power = 64,
                                      folding_para = {'power': 12, 'counts': 0}):
        """Computes interaction fingerprint features for protein-ligand complexes using an avg ecfp algorithm.
        """
        print('Compute ifp features (contacts: [%lf, %lf] Angstrom) for the protein-ligand complex...' % (contact_bin[0], contact_bin[1]))
        idfs = []
        idflist = [[], []]             
        for chain in self.pro:
            print('Processing chain %s and the ligand...' % chain)
            contacts = np.nonzero((self.pairwise_distances[chain] > contact_bin[0]) & (self.pairwise_distances[chain] < contact_bin[1]))
            cont = ([int(c) for c in contacts[0]], [int(c) for c in contacts[1]])
            mols = (self.pro[chain], self.lig)   
            if alg_type == 'avg':
                identifiers = getECFPidentifiers_molpair_avg(mols = mols, 
                                                             heavy_atoms = heavy_atoms,
                                                             base_prop = base_prop,
                                                             sa_dicts = self.sadicts[chain],
                                                             contacts = cont, 
                                                             ifptype = ifptype,
                                                             degrees = ecfp_radius,
                                                             parameters = ch_para,
                                                             hash_type = hash_type,
                                                             idf_power = idf_power)
            elif alg_type == 'classic':
                identifiers = getECFPidentifiers_molpair_classic(mols_info = mols, 
                                                                 prop = base_prop,
                                                                 sa_dicts = self.sadicts[chain],
                                                                 contacts = cont,
                                                                 degrees = ecfp_radius,
                                                                 hash_type = hash_type,
                                                                 idf_power = idf_power,
                                                                 ifptype = ifptype)
            else:
                print('Wrong algorithm type!!!')
                return []
            
            if ifptype in ['ecfp']:
                for i in [0, 1]:
                    idflist[i] += identifiers[i]
            elif ifptype in ['splif', 'plec']:
                idfs += identifiers
            else:
                print('Wrong interaction fingerprint type! Please provide mode for computing ifp!')
                return []

        if ifptype in ['ecfp']:
            ifp = np.concatenate([ifp_folding(identifiers = tmp, 
                                              channel_power = folding_para['power'], 
                                              counts = folding_para['counts']) for tmp in idflist])  
        else:
            ifp = ifp_folding(identifiers = idfs,
                              channel_power = folding_para['power'],
                              counts = folding_para['counts'])
                         
        return ifp
    
    def compute_ifp_features_in_range_mulpr(self,
                                            contact_bin, 
                                            ifptype = 'ecfp', 
                                            alg_type = 'avg',
                                            ch_para = [{'weighted': 0, 'alpha': -1, 'alpha_step': 0.1},
                                                       {'weighted': 0, 'alpha': -1, 'alpha_step': 0.1}],
                                            ecfp_radius = [2, 2],
                                            heavy_atoms = 0,
                                            base_prop = ['AtomicMass'],
                                            hash_type = 'str',
                                            idf_power = 64,
                                            folding_para = {'power': np.arange(6, 16, 1), 'counts': 0}):
        """Computes interaction fingerprint features for protein-ligand complexes using an avg ecfp algorithm.
        """
        print('Compute ifp features (contacts: [%lf, %lf] Angstrom) for the protein-ligand complex...' % (contact_bin[0], contact_bin[1]))
        ifp = {}
        idfs = []
        idflist = [[], []]             
        for chain in self.pro:
            print('Processing chain %s and the ligand...' % chain)
            contacts = np.nonzero((self.pairwise_distances[chain] > contact_bin[0]) & (self.pairwise_distances[chain] < contact_bin[1]))
            cont = ([int(c) for c in contacts[0]], [int(c) for c in contacts[1]])
            mols = (self.pro[chain], self.lig)   
            if alg_type == 'avg':
                identifiers = getECFPidentifiers_molpair_avg(mols = mols, 
                                                             heavy_atoms = heavy_atoms,
                                                             base_prop = base_prop,
                                                             sa_dicts = self.sadicts[chain],
                                                             contacts = cont, 
                                                             ifptype = ifptype,
                                                             degrees = ecfp_radius,
                                                             parameters = ch_para,
                                                             hash_type = hash_type,
                                                             idf_power = idf_power)
            elif alg_type == 'classic':
                identifiers = getECFPidentifiers_molpair_classic(mols_info = mols, 
                                                                 prop = base_prop,
                                                                 sa_dicts = self.sadicts[chain],
                                                                 contacts = cont,
                                                                 degrees = ecfp_radius,
                                                                 hash_type = hash_type,
                                                                 idf_power = idf_power,
                                                                 ifptype = ifptype)
            else:
                print('Wrong algorithm type!!!')
                return []
            
            if ifptype in ['ecfp']:
                for i in [0, 1]:
                    idflist[i] += identifiers[i]
            elif ifptype in ['splif', 'plec']:
                idfs += identifiers
            else:
                print('Wrong interaction fingerprint type! Please provide mode for computing ifp!')
                return []

        if ifptype in ['ecfp']:
            for pr in folding_para['power']:
                ifp[pr] = np.concatenate([ifp_folding(identifiers = tmp, 
                                                      channel_power = pr, 
                                                      counts = folding_para['counts']) for tmp in idflist])  
        else:
            for pr in folding_para['power']:
                ifp[pr] = ifp_folding(identifiers = idfs,
                                  channel_power = pr,
                                  counts = folding_para['counts'])
                         
        return ifp


    def featurize_ifp(self,
                      contact_bins = None,
                      ifptype = 'ecfp', 
                      alg_type = 'avg',
                      ecfp_radius = [2, 2],
                      heavy_atoms = 0,
                      base_prop = ['AtomicMass'],
                      hash_type = 'str',
                      idf_power = 64,
                      folding_para = {'power': 12, 'counts': 0}):
        """Computes IFP featurization of protein-ligand binding pocketn using an avg algorithm.
        """
        print('Construct ifp of type %s......' % ifptype)
        ch_para_pro = {'weighted': self.weighted, 'alpha': self.alpha[0], 'alpha_step': self.alpha_step[0]}
        ch_para_lig = {'weighted': self.weighted, 'alpha': self.alpha[1], 'alpha_step': self.alpha_step[1]}
        ch_para = [ch_para_pro, ch_para_lig]

        if contact_bins is not None:
            self.contact_bins = contact_bins
        ifps = np.array([])
        for contact_bin in self.contact_bins:
            tmp = self.compute_ifp_features_in_range(contact_bin = contact_bin,
                                                     ifptype = ifptype,   
                                                     alg_type = alg_type,
                                                     ch_para = ch_para,
                                                     ecfp_radius = ecfp_radius,
                                                     heavy_atoms = heavy_atoms,
                                                     base_prop = base_prop,
                                                     hash_type = hash_type,
                                                     idf_power = idf_power,
                                                     folding_para = folding_para)

            ifps = np.concatenate((ifps, tmp))
        return ifps

    
    def featurize_ifp_mulpr(self,
                            contact_bins = None,
                            ifptype = 'ecfp', 
                            alg_type = 'avg',
                            ecfp_radius = [2, 2],
                            heavy_atoms = 0,
                            base_prop = ['AtomicMass'],
                            hash_type = 'str',
                            idf_power = 64,
                            folding_para = {'power': np.arange(6, 16, 1), 'counts': 0}):
        """Computes IFP featurization of protein-ligand binding pocketn using an avg algorithm.
        """
        print('Construct ifp of type %s......' % ifptype)
        res = {}
        ch_para_pro = {'weighted': self.weighted, 'alpha': self.alpha[0], 'alpha_step': self.alpha_step[0]}
        ch_para_lig = {'weighted': self.weighted, 'alpha': self.alpha[1], 'alpha_step': self.alpha_step[1]}
        ch_para = [ch_para_pro, ch_para_lig]

        if contact_bins is not None:
            self.contact_bins = contact_bins
        
        for pr in folding_para['power']:
            res[pr] = np.array([])
        for contact_bin in self.contact_bins:
            tmp = self.compute_ifp_features_in_range_mulpr(contact_bin = contact_bin,
                                                           ifptype = ifptype,   
                                                           alg_type = alg_type,
                                                           ch_para = ch_para,
                                                           ecfp_radius = ecfp_radius,
                                                           heavy_atoms = heavy_atoms,
                                                           base_prop = base_prop,
                                                           hash_type = hash_type,
                                                           idf_power = idf_power,
                                                           folding_para = folding_para)
            
            for pr in folding_para['power']:
                res[pr] = np.concatenate((res[pr], tmp[pr]))
        return res
