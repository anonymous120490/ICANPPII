import argparse
import random
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from unimol_tools.data.datahub import DataHub


def set_seed(seed: int = 1) -> None:
    """Fix seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_smiles(smiles: str):
    """Return the canonical SMILES."""

    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except Exception:
        return None


def build_datahub_params(batch_size: int, use_cuda: bool) -> dict:
    """Build the parameter dict passed to ``DataHub``."""
    return {
        "data_type": "molecule",
        "batch_size": batch_size,
        "remove_hs": False,
        "model_name": "unimolv2",
        "model_size": "84m",
        "use_cuda": use_cuda,
        "use_ddp": False,
        "use_gpu": "all",
        "save_path": None,
        "multiprocess": True,
    }


def process_data(csv_path: str, pt_path: str, datahub_params: dict) -> None:
    """Generate UniMol2 inputs for canonicalized SMILES and save them to a .pt file.

    Writes a dict {canonical_smiles: unimol_input}; SMILES that RDKit cannot
    parse are dropped.

    Args:
        csv_path: CSV path containing a can_smi column.
        pt_path: Save path for the .pt file containing the input dict.
        datahub_params: Keyword params for DataHub.
    """
    df = pd.read_csv(csv_path)
    smiles_list = df["can_smi"].unique()
    smiles_list = [canonical_smiles(s) for s in smiles_list]
    smiles_list = pd.DataFrame(smiles_list).dropna().values.flatten()
    datahub = DataHub(data=smiles_list, task="repr", is_train=False, **datahub_params)
    unimol_inputs = {smi: datahub.data["unimol_input"][i] for i, smi in enumerate(smiles_list)}
    torch.save(unimol_inputs, pt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output .pt file.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")

    datahub_params = build_datahub_params(args.batch_size, use_cuda)
    process_data(args.input, args.output, datahub_params)


if __name__ == "__main__":
    main()