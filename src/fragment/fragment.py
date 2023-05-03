"""Recusively Fragment a molecule"""

from rdkit import Chem


CHI_TETRAHEDRAL_CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
CHI_TETRAHEDRAL_CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW


def fragment_on_bond(mol, atom1, atom2):
    bond = mol.GetBondBetweenAtoms(atom1, atom2)
    new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(0, 0)])
    # Chem.SanitizeMol(new_mol)
    return new_mol


def parity_shell(values):
    values = list(values)
    N = len(values)
    num_swaps = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            if values[i] > values[j]:
                values[i], values[j] = values[j], values[i]
                num_swaps += 1
    return num_swaps % 2


def get_bond_parity(mol, atom_id):
    """Compute the parity of the atom's bond permutation

    Return None if it does not have tetrahedral chirality,
    0 for even parity, or 1 for odd parity.
    """
    atom_obj = mol.GetAtomWithIdx(atom_id)

    # Return None unless it has tetrahedral chirality
    chiral_tag = atom_obj.GetChiralTag()
    if chiral_tag not in (CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW):
        return None

    # Get the list of atom ids for the each atom it's bonded to.
    other_atom_ids = [bond.GetOtherAtomIdx(atom_id) for bond in atom_obj.GetBonds()]

    # Use those ids to determine the parity
    return parity_shell(other_atom_ids)


def set_bond_parity(mol, atom_id, old_parity, old_other_atom_id, new_other_atom_id):
    """Compute the new bond parity and flip chirality if needed to match the old parity"""

    atom_obj = mol.GetAtomWithIdx(atom_id)
    # Get the list of atom ids for the each atom it's bonded to.
    other_atom_ids = [bond.GetOtherAtomIdx(atom_id) for bond in atom_obj.GetBonds()]

    # Replace id from the new wildcard atom with the id of the original atom
    i = other_atom_ids.index(new_other_atom_id)
    other_atom_ids[i] = old_other_atom_id

    # Use those ids to determine the parity
    new_parity = parity_shell(other_atom_ids)
    if old_parity != new_parity:
        # If the parity has changed, invert the chirality
        atom_obj.InvertChirality()


def fragment_chiral(mol, atom1, atom2):
    """Cut the bond between atom1 and atom2 and replace with connections to wildcard atoms
    Return the fragmented structure as a new molecule.
    """
    rwmol = Chem.RWMol(mol)

    atom1_parity = get_bond_parity(mol, atom1)
    atom2_parity = get_bond_parity(mol, atom2)

    rwmol.RemoveBond(atom1, atom2)
    wildcard1 = rwmol.AddAtom(Chem.Atom(0))
    wildcard2 = rwmol.AddAtom(Chem.Atom(0))
    new_bond1 = rwmol.AddBond(atom1, wildcard1, Chem.BondType.SINGLE)
    new_bond2 = rwmol.AddBond(atom2, wildcard2, Chem.BondType.SINGLE)

    if atom1_parity is not None:
        set_bond_parity(rwmol, atom1, atom1_parity, atom2, wildcard1)
    if atom2_parity is not None:
        set_bond_parity(rwmol, atom2, atom2_parity, atom1, wildcard2)

    new_mol = rwmol.GetMol()
    new_mol.ClearComputedProps()
    # Chem.SanitizeMol(new_mol)
    return new_mol


def frag_rec(
    smiles,
    pattern="[!#0;!#1]-!@[!#0;!#1]",
    fragments=None,
    depth=0,
    max_depth=5,
    visited=None,
):

    if depth > max_depth:
        return []

    if fragments is None:
        fragments = set()

    if visited is None:
        visited = set()

    pattern = Chem.MolFromSmarts(pattern)
    mol = Chem.MolFromSmiles(smiles)
    matches = mol.GetSubstructMatches(pattern)

    for a1, a2 in matches:
        mol_chiral = fragment_chiral(mol, a1, a2)
        smiles_chiral = Chem.MolToSmiles(mol_chiral, isomericSmiles=True)
        mol_on_bond = fragment_on_bond(mol, a1, a2)
        smiles_on_bond = Chem.MolToSmiles(mol_on_bond, isomericSmiles=True)
        _frags = set(smiles_chiral.split(".") + smiles_on_bond.split("."))
        fragments.update(_frags)
        for _frags_smi in _frags:
            if _frags_smi not in visited:
                visited.add(_frags_smi)
                fragments.update(
                    frag_rec(_frags_smi, fragments, depth + 1, max_depth, visited)
                )
    return list(fragments)
