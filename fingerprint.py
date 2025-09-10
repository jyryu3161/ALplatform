from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import __version__ as rdkit_version

def get_morgan_fingerprint_safe(mol, radius=2, nBits=1024):
    """RDKit 버전에 따라 안전하게 Morgan fingerprint를 생성하는 함수"""
    try:
        major, minor, patch = map(int, rdkit_version.split('.'))
        if major >= 2023 and minor >= 9:
            try:
                from rdkit.Chem import rdMolDescriptors
                generator = rdMolDescriptors.GetMorganGenerator(radius=radius)
                fp = generator.GetFingerprint(mol)
                bit_fp = DataStructs.CreateFromBitString('0' * nBits)
                DataStructs.ConvertToNumpyArray(fp, bit_fp)
                return bit_fp
            except:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        else:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    except:
        try:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        except:
            return None