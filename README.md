# IFPscore
IFP-based scoring functions

1. Requirements
   IFPscore currently supports a Linux system and Python 3.6, and requires main dependency packages as follows. 
   - deemchem (https://github.com/deepchem/deepchem)
   - rdkit (https://www.rdkit.org/)
   - numpy (https://numpy.org/)						
   - sklearn (https://scikit-learn.org/stable/)
   - multiprocessing (https://docs.python.org/3/library/multiprocessing.html)
   - Bio (https://biopython.org/)
   - scipy (https://www.scipy.org/)

2. Data downloading and preprocessing
	1) Downloading:
	
	   Model construction - 'PDBbind_refined' data folder: PDBbind refined set (http://www.pdbbind.org.cn/)
	   
	   validation - 'PDBbind_core' data folder: PDBbind core set (http://www.pdbbind.org.cn/)
	   
				  - 'csarhiqS1' data folder: CSAR-HiQ sets 1 (http://www.csardock.org/)
				  
				  - 'csarhiqS2' data folder: CSAR-HiQ sets 2 (http://www.csardock.org/)
				  
				  - 'csarhiqS3' data folder: CSAR-HiQ sets 3 (http://www.csardock.org/)

	2) Preprocessing:
	
	   PDBbind refined/core sets: save the ligand files as PDB files (e.g. using software like UCSF Chimera)
	   
	   CSAR-HiQ sets: save the protein and ligand in each complex as PDB files (e.g. using software like UCSF Chimera), 
	   		  and name these files as those in PDBbind refined/core sets (e.g. 1ax1_protein.pdb, 1ax1_ligand.pdb in '1ax1' folder)
			  
	3) Put these folders together:
	
	   Create a folder (e.g. 'Score') and put all these data folders (e.g. 'PDBbind_refined', 'PDBbind_core', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3') 
	   
	   and the index folder ('indexes' folder in this repository) in 'Score'
	   
3. Example codes are provided in the 'Examples' folder in this repository
   1) PrtCmmIFPScore - Constructing a PrtCmm IFP Score on PDBbind refined set (excluding the validation sets) and
   						         validating it on the four validation sets (PDBbind core set and CSAR-HiQ sets) using 
						           Pearson's correlation and RMSE
   2) RFprtcmmScore - Constructing an RF-SCORE (PrtCmm version) on PDBbind refined set (excluding the validation sets) and
   						        validating it on the four validation sets (PDBbind core set and CSAR-HiQ sets) using 
						          Pearson's correlation and RMSE
