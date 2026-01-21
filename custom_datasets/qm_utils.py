import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

_ATOMMAP_RE = re.compile(r":(\d+)\]")  # matches [C:12], [nH:7], etc.


def _build_mol_xyz(symbols, pos, charge=0):
    """RDKit mol with atoms in dataset order + conformer from pos + bonds inferred."""
    symbols = list(symbols)
    pos = np.asarray(pos, dtype=np.float64)

    rw = Chem.RWMol()
    for s in symbols:
        rw.AddAtom(Chem.Atom(s))
    mol = rw.GetMol()

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(pos):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    # Infer bonds from geometry (can fail rarely)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    Chem.SanitizeMol(mol)
    return mol


def _build_mol_smiles(smiles):
    """RDKit mol from SMILES with explicit H."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol, addCoords=False)
    Chem.SanitizeMol(mol)
    return mol


def _candidate_matches_smiles_to_xyz(mol_smi, mol_xyz, max_matches=64):
    """
    Return a list of candidate mappings: list of tuples (smi_to_xyz, score)
    where smi_to_xyz is a tuple mapping smi atom idx -> xyz atom idx.
    """
    # We want an isomorphism between graphs. Often either direction works; this tends to:
    matches = mol_xyz.GetSubstructMatches(mol_smi, uniquify=False, maxMatches=max_matches)
    if not matches:
        # Try opposite direction (depending on how RDKit sees query/target)
        matches = mol_smi.GetSubstructMatches(mol_xyz, uniquify=False, maxMatches=max_matches)
        # If this branch hits, we'd need to invert mapping. We'll just handle by returning empty:
        return []

    # Score each match by how well hydrogens sit near their heavy neighbors in xyz coords.
    # Lower score is better.
    conf = mol_xyz.GetConformer()
    xyz_pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_xyz.GetNumAtoms())], dtype=np.float64)

    smi_neighbors = [ [nbr.GetIdx() for nbr in mol_smi.GetAtomWithIdx(i).GetNeighbors()]
                      for i in range(mol_smi.GetNumAtoms()) ]

    results = []
    for smi_to_xyz in matches:
        # smi_to_xyz is a tuple of length mol_smi.GetNumAtoms()
        score = 0.0
        # Hydrogen proximity term: H should be close to its (mapped) neighbor heavy atom(s)
        for i in range(mol_smi.GetNumAtoms()):
            atom = mol_smi.GetAtomWithIdx(i)
            if atom.GetSymbol() != "H":
                continue
            xi = xyz_pos[smi_to_xyz[i]]
            # hydrogen in SMILES has exactly one neighbor in explicit-H graph
            nbs = smi_neighbors[i]
            if not nbs:
                continue
            j = nbs[0]
            xj = xyz_pos[smi_to_xyz[j]]
            score += float(np.sum((xi - xj) ** 2))
        results.append((smi_to_xyz, score))

    results.sort(key=lambda t: t[1])
    return results


def map_smiles_atoms_to_xyz_indices(symbols, pos, smiles, charge=0, max_matches=64):
    """
    Returns:
      smi_to_xyz: tuple mapping atom idx in mol_smi -> atom idx in dataset order
      mol_smi, mol_xyz
    Raises if mapping fails.
    """
    mol_xyz = _build_mol_xyz(symbols, pos, charge=charge)
    mol_smi = _build_mol_smiles(smiles)

    if mol_smi.GetNumAtoms() != mol_xyz.GetNumAtoms():
        raise ValueError(
            f"Atom count mismatch: smiles+Hs has {mol_smi.GetNumAtoms()}, xyz has {mol_xyz.GetNumAtoms()}.\n"
            f"SMILES: {smiles}"
        )

    cands = _candidate_matches_smiles_to_xyz(mol_smi, mol_xyz, max_matches=max_matches)
    if not cands:
        raise ValueError(f"Failed to find substructure/isomorphism match between SMILES and XYZ graphs. SMILES={smiles}")

    smi_to_xyz, best_score = cands[0]
    return smi_to_xyz, mol_smi, mol_xyz


def canonical_smiles_atom_order(mol_smi):
    """
    Get atom indices in the order they appear in RDKit canonical SMILES.
    Uses atom-map numbers to recover the appearance order.
    """
    # Set atom-map numbers = atom index in mol_smi
    for i, a in enumerate(mol_smi.GetAtoms()):
        a.SetAtomMapNum(i)

    smi = Chem.MolToSmiles(mol_smi, canonical=True)

    # Extract map numbers in SMILES appearance order
    order = [int(m.group(1)) for m in _ATOMMAP_RE.finditer(smi)]

    # Cleanup atom-map numbers (optional)
    for a in mol_smi.GetAtoms():
        a.SetAtomMapNum(0)

    # RDKit SMILES may repeat bracketed atoms in some contexts; keep first occurrence order
    seen = set()
    uniq = []
    for idx in order:
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)
    return uniq


def reorder_like_branching_flows(symbols, pos, smiles, charge=0):
    """
    Full preprocessing described:
      1) Heavy atoms in canonical SMILES order (mapped onto dataset atom indices).
      2) Each H inserted immediately before its nearest heavy atom (by Euclidean distance),
         with per-heavy H sorted by distance.

    Returns: (new_symbols, new_pos)
    """
    symbols = list(symbols)
    pos = np.asarray(pos, dtype=np.float32)

    smi_to_xyz, mol_smi, _mol_xyz = map_smiles_atoms_to_xyz_indices(symbols, pos, smiles, charge=charge)

    smi_order = canonical_smiles_atom_order(mol_smi)

    # Convert SMILES appearance order -> dataset indices
    xyz_order_all = [smi_to_xyz[i] for i in smi_order]

    heavy_set = {i for i, s in enumerate(symbols) if s != "H"}
    heavy_order = [i for i in xyz_order_all if i in heavy_set]

    # Fallback if something weird happens
    if len(heavy_order) != len(heavy_set):
        # ensure we at least include all heavy atoms exactly once
        missing = [i for i in heavy_set if i not in set(heavy_order)]
        heavy_order = heavy_order + sorted(missing)

    hyd_idx = [i for i, s in enumerate(symbols) if s == "H"]

    # Hydrogen redistribution: nearest heavy in *space*, inserted before that heavy in the heavy_order sequence
    if hyd_idx and heavy_order:
        heavy_pos = pos[heavy_order]  # (Hn,3)
        hyd_pos = pos[hyd_idx]        # (Kn,3)

        d2 = ((hyd_pos[:, None, :] - heavy_pos[None, :, :]) ** 2).sum(-1)  # (Kn,Hn)
        nearest_slot = d2.argmin(1)
        nearest_d2 = d2[np.arange(len(hyd_idx)), nearest_slot]

        buckets = [[] for _ in range(len(heavy_order))]
        for k, slot in enumerate(nearest_slot):
            buckets[int(slot)].append((float(nearest_d2[k]), hyd_idx[k]))
        for slot in range(len(buckets)):
            buckets[slot].sort(key=lambda t: t[0])
    else:
        buckets = [[] for _ in range(len(heavy_order))]

    new_order = []
    for slot, heavy_i in enumerate(heavy_order):
        new_order.extend([h_i for _, h_i in buckets[slot]])
        new_order.append(heavy_i)

    new_symbols = [symbols[i] for i in new_order]
    new_pos = pos[new_order]
    return new_symbols, new_pos
