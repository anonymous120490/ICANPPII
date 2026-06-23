import argparse
import numpy as np
import pandas as pd
import torch


def collect_unique_ids(df: pd.DataFrame) -> list:
    """Collect unique UniProt ids from input dataFrame."""

    ids = pd.concat([df["uniprot_id1"], df["uniprot_id2"]], ignore_index=True)
    ids = ids.dropna().drop_duplicates()
    return ids.tolist()


def infer_zero_vector(precomputed: dict) -> np.ndarray:
    """Return a zero vector matching the shape/dtype of any precomputed embedding."""

    sample = next(iter(precomputed.values()))
    sample = np.asarray(sample)
    return np.zeros_like(sample)


def build_kg_embeddings(uniprot_ids: list, precomputed: dict) -> dict:
    """Look up KG embeddings for each uniprot id, zero-filling those not precomputed.

    Args:
        uniprot_ids: list of UniProt IDs.
        precomputed: Dict {uniprot_id: embedding}.

    Returns:
        dict[str, np.ndarray]: Maps uniprot ID to its embedding, or a zero vector if the ID was not in precomputed.
    """

    zero_vector = infer_zero_vector(precomputed)
    embeddings = {}
    n_found = 0
    for pid in uniprot_ids:
        if pid in precomputed:
            embeddings[pid] = precomputed[pid]
            n_found += 1
        else:
            embeddings[pid] = zero_vector.copy()
    print(f"Found {n_found}/{len(uniprot_ids)} ids in precomputed KG embeddings; "
          f"{len(uniprot_ids) - n_found} assigned zero vectors.")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output .pt file path.")
    parser.add_argument("--precomputed", default="data/features/precomputed_kg_embedding.pt", 
                        help="Precomputed KG embedding .pt file (default: data/features/precomputed_kg_embedding.pt).")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    uniprot_ids = collect_unique_ids(df)
    print(f"Collected {len(uniprot_ids)} unique proteins from {args.input}")

    precomputed = torch.load(args.precomputed, map_location="cpu")
    embeddings = build_kg_embeddings(uniprot_ids, precomputed)

    torch.save(embeddings, args.output)
    print(f"Saved {len(embeddings)} KG embeddings to {args.output}")


if __name__ == "__main__":
    main()
