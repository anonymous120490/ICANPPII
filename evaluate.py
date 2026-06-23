import argparse
import os
import numpy as np
import torch
import config
from src.utils import (
    predict, performance_evaluation_proba, compute_uncertainty,
    compute_calibration_errors, build_unimol_encoder, build_collate_fn,
    build_model, load_dataloader, set_seed, get_device,
)

METRIC_KEYS = ["accuracy", "precision", "recall", "specificity", "f1_score", "auc", "aupr"]

ENSEMBLE_THRESHOLD = 0.5


def print_metrics(results, keys=METRIC_KEYS):
    print("  " + "  ".join(f"{k}: {results[k]:.4f}" for k in keys))


def _proba(model, dataloader, device):
    """Return (labels, positive-class probabilities) for one model."""

    labels, logits = predict(model, dataloader, device)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return labels, probs


def run_split(split, collate_fn, device, seeds):
    print(f"\n========== {split} split | seeds {seeds} ==========")

    loaders, test_df = load_dataloader(split, collate_fn, device, which=("valid", "test"), return_test_df=True)
    valid_dl, test_dl = loaders["valid"], loaders["test"]

    os.makedirs(config.output_dir(split), exist_ok=True)

    test_probs = []
    valid_labels = test_labels = None

    for seed in seeds:
        ckpt = config.model_path(split, seed)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Missing checkpoint for seed {seed}: {ckpt}. "
                f"Train it first (python train.py --splits {split} --seed {seed}).")
        model = build_model(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded model (seed {seed}): {ckpt}")

        valid_labels, vp = _proba(model, valid_dl, device)
        test_labels, tp = _proba(model, test_dl, device)
        test_probs.append(tp)

        # Per-seed prediction CSV
        seed_threshold = performance_evaluation_proba(valid_labels, vp)[1]["optimal_threshold"] # The threshold is chosen on this seed's validation.
        seed_preds, _ = performance_evaluation_proba(test_labels, tp, seed_threshold)
        seed_df = test_df.copy()
        seed_df["pred_label"] = seed_preds
        seed_df["pred_prob"] = tp
        seed_out = config.prediction_path(split, seed)
        seed_df.to_csv(seed_out, index=False)
        print(f"  per-seed predictions saved to: {seed_out}")

    test_arr = np.stack(test_probs, axis=0)     # [n_seeds, n_test]

    # Ensemble-averaged probabilities, split at the fixed 0.5 threshold.
    unc = compute_uncertainty(test_arr)
    predictions, test_results = performance_evaluation_proba(test_labels, unc["mean_prob"], ENSEMBLE_THRESHOLD)
    ece, mce = compute_calibration_errors(unc["mean_prob"], test_labels)

    print(f"[test | threshold={ENSEMBLE_THRESHOLD}]")
    print_metrics(test_results)
    print(f"  ECE: {ece:.4f}  MCE: {mce:.4f}")

    # Save predictions with uncertainties.
    test_df = test_df.copy()
    test_df["pred_label"] = predictions
    test_df["pred_prob"] = unc["mean_prob"]
    test_df["aleatoric"] = unc["aleatoric"]
    test_df["epistemic"] = unc["epistemic"]
    test_df["total_uncertainty"] = unc["total"]
    test_df["threshold"] = ENSEMBLE_THRESHOLD

    out_path = config.ensemble_prediction_path(split)
    test_df.to_csv(out_path, index=False)
    print(f"Ensemble predictions saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=config.SPLITS, choices=config.SPLITS, help="Dataset splitting strategies.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Seeds for the ensemble models.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    device = get_device(args.device)
    set_seed(args.seeds[0])
    print(f"Device: {device} | ensemble seeds: {args.seeds}")

    collate_fn = build_collate_fn(build_unimol_encoder(device))

    for split in args.splits:
        run_split(split, collate_fn, device, args.seeds)


if __name__ == "__main__":
    main()
