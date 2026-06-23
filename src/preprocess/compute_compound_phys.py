import argparse
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAromaticCarbocycles, CalcNumAromaticRings, CalcNumAliphaticRings, CalcNumAromaticHeterocycles, CalcNumHeteroatoms, CalcNumSaturatedHeterocycles, CalcNumSaturatedCarbocycles, CalcNumSaturatedRings
from rdkit.Chem.Lipinski import NOCount

RDLogger.DisableLog("rdApp.*")

DESCRIPTORS = [CalcNumRings, CalcNumAromaticRings, CalcNumAromaticCarbocycles, CalcNumAromaticHeterocycles, CalcNumAliphaticRings, CalcNumSaturatedRings,
               CalcNumSaturatedCarbocycles, CalcNumSaturatedHeterocycles, CalcNumHeteroatoms, NOCount]

def compute_compound_props(smiles_list):
    """Compute RDKit descriptors for a list of SMILES.

    Args:
        smiles_list: a list of SMILES strings.

    Returns:
        dict[str, np.ndarray]: Maps each SMILES string to a property array of shape (len(DESCRIPTORS),) containing the computed descriptors.
    """
    props_dict = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[skip] invalid SMILES: {smi}")
            continue
        props_dict[smi] = np.array([fn(mol) for fn in DESCRIPTORS], dtype=np.float32)
    return props_dict

def main():
    parser = argparse.ArgumentParser(description="Precompute compound physicochemical descriptors.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output .pt file path.")
    parser.add_argument("--smiles-col", default="can_smi", help="SMILES column name (default: can_smi).")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    smiles_list = df[args.smiles_col].dropna().unique().tolist()
    props = compute_compound_props(smiles_list)
    props = {k: torch.from_numpy(v) for k, v in props.items()}
    torch.save(props, args.output)

if __name__ == "__main__":
    main()