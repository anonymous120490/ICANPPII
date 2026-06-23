import argparse
import pandas as pd
import esm
import torch
from tqdm import tqdm

REPR_LAYER = 30

def build_unique_proteins(df: pd.DataFrame) -> dict:
    """Collect unique uniprot id and corresponding protein sequences from input dataframe.

    Args:
        df: dataframe with uniprot_id1, uniprot_id2, seq1, seq2 columns.
    
    Returns:
        dict[str, str]: Maps each UniProt ID to its protein sequence.
    """

    proteins = pd.DataFrame({
        "uniprot_id": pd.concat([df["uniprot_id1"], df["uniprot_id2"]], ignore_index=True),
        "sequence": pd.concat([df["seq1"], df["seq2"]], ignore_index=True),
    })
    proteins = proteins.dropna(subset=["uniprot_id", "sequence"])
    proteins = proteins.drop_duplicates(subset="uniprot_id")
    return dict(zip(proteins["uniprot_id"], proteins["sequence"]))

def compute_embeddings(seqs: dict, device: torch.device) -> dict:
    """Compute per-residue ESM-2 650M embeddings for each protein sequence through the pretrained esm2_t33_650M_UR50D model.

    Args:
        seqs: Dict mapping each UniProt ID to its amino-acid sequence.
        device: Device used to run the model.

    Returns:
        dict[str, torch.Tensor]: Maps each UniProt ID to a CPU tensor of shape [L, 1280].
    """

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    representations = {}
    for pid, seq in tqdm(seqs.items(), desc="ESM-2"):
        batch_tokens = torch.LongTensor(alphabet.encode(seq)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[REPR_LAYER], return_contacts=False)
        representations[pid] = out["representations"][REPR_LAYER].squeeze(0).detach().cpu()
    return representations

def main():
    parser = argparse.ArgumentParser(description="Precompute ESM-2 650M protein embeddings from a dataset CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output .pt file path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    device = torch.device(args.device)
    df = pd.read_csv(args.input)
    seqs = build_unique_proteins(df)
    print(f"Collected {len(seqs)} unique proteins from {args.input}")

    representations = compute_embeddings(seqs, device)
    torch.save(representations, args.output)
    print(f"Saved {len(representations)} embeddings to {args.output}")

if __name__ == "__main__":
    main()    