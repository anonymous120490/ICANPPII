import os
import argparse
import copy
import time
import pandas as pd
import torch
import config
from src.utils import (
    train, predict, performance_evaluation,
    build_unimol_encoder, build_collate_fn, load_model, load_dataloader,
    set_seed, get_device,
)

METRIC_KEYS = ["accuracy", "precision", "recall", "specificity", "f1_score", "auc", "aupr"]


def print_metrics(results, keys=METRIC_KEYS):
    print("  " + "  ".join(f"{k}: {results[k]:.4f}" for k in keys))


def _result_row(epoch_type, epoch, split, seed, results, infer_time, train_hours):
    return {
        "model": "ICAN-PPII",
        "epoch_type": epoch_type,
        "epoch": epoch,
        "split": split,
        "seed": seed,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "total_training_hours": train_hours,
        "inference_time_seconds": infer_time,
        "roc_auc": results["auc"],
        "aupr": results["aupr"],
        "precision": results["precision"],
        "accuracy": results["accuracy"],
        "recall": results["recall"],
        "f1": results["f1_score"],
        "specificity": results["specificity"],
    }


def run_split(split, collate_fn, device, seed, num_epochs):
    print(f"\n========== {split} split ==========")
    train_start = time.time()

    loaders = load_dataloader(split, collate_fn, device)
    train_dl, valid_dl, test_dl = loaders["train"], loaders["valid"], loaders["test"]
    model, criterion, optimizer = load_model(device)

    best_model = None
    best_auc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")

        label, pred, _ = train(model, train_dl, optimizer, criterion, device)
        _, tr_results = performance_evaluation(label, pred)
        print("[train]")
        print_metrics(tr_results)

        label, pred = predict(model, valid_dl, device)
        _, val_results = performance_evaluation(label, pred)
        print("[valid]")
        print_metrics(val_results)

        if val_results["auc"] > best_auc:
            best_model = copy.deepcopy(model)
            best_auc = val_results["auc"]
            best_epoch = epoch
            print(f"  -> best AUC improved at epoch {best_epoch}: {best_auc:.4f}")

    train_hours = (time.time() - train_start) / 3600

    # Last-epoch test
    print("\n[test | last epoch]")
    t0 = time.time()
    label, pred = predict(model, test_dl, device)
    _, last_results = performance_evaluation(label, pred)
    last_time = time.time() - t0
    print_metrics(last_results)

    # Best-epoch test
    print(f"\n[test | best epoch {best_epoch}]")
    t0 = time.time()
    label, pred = predict(best_model, test_dl, device)
    _, best_results = performance_evaluation(label, pred)
    best_time = time.time() - t0
    print_metrics(best_results)

    # Save results CSV + best model
    os.makedirs(config.output_dir(split), exist_ok=True)
    results_data = [
        _result_row("last_epoch", num_epochs, split, seed, last_results, last_time, train_hours),
        _result_row("best_epoch", best_epoch, split, seed, best_results, best_time, train_hours),
    ]
    csv_path = config.results_path(split, seed)
    pd.DataFrame(results_data).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    model_path = config.model_path(split, seed)
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=config.SPLITS, choices=config.SPLITS, help="Dataset splitting strategies.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Seeds for the ensemble models.")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Training epochs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device} | seeds: {args.seeds} | epochs: {args.epochs}")

    # One UniMol2 encoder instance is enough to build the collate function.
    collate_fn = build_collate_fn(build_unimol_encoder(device))

    for seed in args.seeds:
        set_seed(seed)
        for split in args.splits:
            run_split(split, collate_fn, device, seed, args.epochs)


if __name__ == "__main__":
    main()
