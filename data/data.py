"""
    File to load dataset based on user control from main file
"""
from data.molecules import MoleculeDataset
from data.financial import FinancialDatapack

def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)
    elif DATASET_NAME == 'nasdaq100':
        return FinancialDatapack(DATASET_NAME)
    