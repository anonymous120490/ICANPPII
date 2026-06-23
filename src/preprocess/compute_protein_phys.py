import argparse
import os
import subprocess
import numpy as np
import pandas as pd
import torch


PCP_COLUMNS = ["PCP_PC", "PCP_NC", "PCP_NE", "PCP_PO", "PCP_NP", "PCP_AL", "PCP_CY",
    "PCP_AR", "PCP_AC", "PCP_BS", "PCP_NE_pH", "PCP_HB", "PCP_HL", "PCP_NT",
    "PCP_HX", "PCP_SC", "PCP_TN", "PCP_SM", "PCP_LR"]


def build_unique_proteins(df: pd.DataFrame) -> pd.DataFrame:
    """Collect unique (uniprot_id, sequence) protein rows from both PPI partners.

    Args:
        df: dataframe with uniprot_id1, uniprot_id2, seq1, seq2 columns.

    Returns:
        DataFrame with uniprot_id and sequence columns.
    """

    proteins = pd.DataFrame({"uniprot_id": pd.concat([df["uniprot_id1"], df["uniprot_id2"]], ignore_index=True),
                             "sequence": pd.concat([df["seq1"], df["seq2"]], ignore_index=True)})
    return proteins.drop_duplicates().reset_index(drop=True)


def write_fasta(proteins: pd.DataFrame, fasta_path: str) -> None:
    """Write proteins to a FASTA file.

    Args:
        proteins: DataFrame with uniprot_id and sequence columns.
        fasta_path: Output FASTA file path.
    """
    with open(fasta_path, "w") as f:
        for row in proteins.itertuples(index=False):
            f.write(f">{row.uniprot_id}\n{row.sequence}\n\n")


def run_pfeature_pcp(fasta_path: str, raw_csv_path: str) -> None:
    """Compute physicochemical properties (PCP) descriptors via Pfeature's pfeature_comp.

    Args:
        fasta_path: Input FASTA file path.
        raw_csv_path: Output path for the generated PCP descriptor CSV file.
    """

    subprocess.run(["pfeature_comp", "-i", fasta_path, "-o", raw_csv_path, "-j", "PCP"], check=True)


def compute_protein_props(df: pd.DataFrame, output_path: str, work_dir: str) -> dict:
    """Compute and save Pfeature PCP descriptors.

    Args:
        df: dataframe with uniprot_id1,uniprot_id2, seq1, seq2 columns.
        output_path: Output path for the .pt file storing the descriptor dict.
        work_dir: Directory for intermediate files. If omitted, the directory of output_path is used. if output_path is None, the current directory is used.

    Returns:
        dict[str, np.ndarray]: Maps each UniProt ID to a descriptor array. Also saved to output_path.
    """

    work_dir = work_dir or os.path.dirname(output_path) or "."
    os.makedirs(work_dir, exist_ok=True)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fasta_path = os.path.join(work_dir, "protein.fasta")
    raw_csv_path = os.path.join(work_dir, "protein_phy_raw.csv")

    proteins = build_unique_proteins(df)
    write_fasta(proteins, fasta_path)
    run_pfeature_pcp(fasta_path, raw_csv_path)

    protein_phy = pd.read_csv(raw_csv_path)
    features = protein_phy[PCP_COLUMNS].to_numpy(dtype=np.float32)

    props = {uid: features[i] for i, uid in enumerate(proteins["uniprot_id"].values)}
    torch.save(props, output_path)
    return props


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output .pt file path.")
    parser.add_argument("--work-dir", default=None, help="Directory for intermediate files (default: output dir).")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    compute_protein_props(df, output_path=args.output, work_dir=args.work_dir)

if __name__ == "__main__":
    main()