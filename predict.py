import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import config
from src.dataset import PPIInhibitorDataset
from src.utils import (
    predict, compute_uncertainty, build_unimol_encoder, build_collate_fn,
    build_model, set_seed, get_device,
)


ENSEMBLE_THRESHOLD = 0.5


def _proba(model, dataloader, device):
    """Return predicted probabilities"""

    _, logits = predict(model, dataloader, device)
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def _member_path(output, ckpt):
    """Build a per-checkpoint output path."""

    root, ext = os.path.splitext(output)
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    return f"{root}_{stem}{ext}"


def load_inference_dataloader(input_csv, collate_fn, device):
    """Build a DataLoader and the processed dataframe from an input CSV."""
    
    dataset = PPIInhibitorDataset(
        input_csv, config.ESM2_PATH, config.PROT_PHY_PATH, config.KG_PATH,
        config.COMP_PHY_PATH, config.UNIMOL2_PATH, device)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        drop_last=False, collate_fn=collate_fn)
    return loader, dataset.get_df()


def run_inference(input_csv, output, checkpoints, collate_fn, device):
    print(f"\n========== inference | {len(checkpoints)} checkpoint(s) ==========")

    loader, base_df = load_inference_dataloader(input_csv, collate_fn, device)
    print(f"Loaded {len(base_df)} samples from {input_csv}")

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    test_probs = []
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model = build_model(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded model: {ckpt}")

        tp = _proba(model, loader, device)
        test_probs.append(tp)

        # Per-checkpoint prediction CSV (fixed 0.5 threshold; no labels required).
        member_df = base_df.copy()
        member_df["pred_label"] = (tp >= ENSEMBLE_THRESHOLD).astype(int)
        member_df["pred_prob"] = tp
        member_df["threshold"] = ENSEMBLE_THRESHOLD
        member_out = _member_path(output, ckpt)
        member_df.to_csv(member_out, index=False)
        print(f"  per-checkpoint predictions saved to: {member_out}")

    test_arr = np.stack(test_probs, axis=0)     # [n_checkpoints, n_samples]

    # Ensemble-averaged probabilities, split at the fixed 0.5 threshold.
    unc = compute_uncertainty(test_arr)
    predictions = (unc["mean_prob"] >= ENSEMBLE_THRESHOLD).astype(int)

    out_df = base_df.copy()
    out_df["pred_label"] = predictions
    out_df["pred_prob"] = unc["mean_prob"]
    out_df["aleatoric"] = unc["aleatoric"]
    out_df["epistemic"] = unc["epistemic"]
    out_df["total_uncertainty"] = unc["total"]
    out_df["threshold"] = ENSEMBLE_THRESHOLD
    out_df.to_csv(output, index=False)
    print(f"Ensemble predictions saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Ensemble output CSV path. Per-checkpoint CSVs are written alongside it with a '_<checkpoint name>' suffix.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Trained model checkpoint (.pt) paths. If multiple, their averaged probabilities give the ensemble prediction.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    device = get_device(args.device)
    set_seed(1)
    print(f"Device: {device} | {len(args.checkpoints)} checkpoint(s)")

    collate_fn = build_collate_fn(build_unimol_encoder(device))
    run_inference(args.input, args.output, args.checkpoints, collate_fn, device)


if __name__ == "__main__":
    main()
