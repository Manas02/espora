from rdkit import Chem
from rdkit.Chem import Descriptors


def get_mw(smi):
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.MolWt(mol)