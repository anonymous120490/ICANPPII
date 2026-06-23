import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score, roc_curve,
)
import config
from src.dataset import PPIInhibitorDataset
from src.model import PPIInhibitorModel


# Reproducibility / device
def set_seed(seed):
    """Fix seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device):
    use_cuda = device == "cuda" and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


# Train / Predict
def train(model, dataloader, optimizer, criterion, device):
    """Run one training epoch and return its predictions and mean loss."""

    model.train()
    preds = []
    labels = []
    losses = 0

    progress_bar = tqdm(dataloader, desc="Train", ncols=80, bar_format='{desc:<8} |{bar:30}| {percentage:3.0f}% ({n_fmt}/{total_fmt})')
    for batch in progress_bar:
        unimol2 = {k: v.to(device) for k, v in batch[0][0].items()}
        esm1, esm2, auxiliary, valid_length, label = [x.to(device) for x in batch[1:]]
        pred = model(unimol2, esm1, esm2, auxiliary, valid_length).squeeze()

        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())

        loss = criterion(pred, label)
        losses += loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss:\t{}'.format(losses / len(dataloader)))

    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    mean_loss = losses / len(dataloader)
    return labels, preds, mean_loss


def predict(model, dataloader, device):
    """Run inference and return predictions and labels."""

    model.eval()
    preds = []
    labels = []

    progress_bar = tqdm(dataloader, desc="Predict", ncols=80, bar_format='{desc:<8} |{bar:30}| {percentage:3.0f}% ({n_fmt}/{total_fmt})')
    with torch.no_grad():
        for batch in progress_bar:
            unimol2 = {k: v.to(device) for k, v in batch[0][0].items()}
            esm1, esm2, auxiliary, valid_length, label = [x.to(device) for x in batch[1:]]
            pred = model(unimol2, esm1, esm2, auxiliary, valid_length).squeeze()
            preds.append(pred.detach().cpu())
            labels.append(label.detach().cpu())

    preds = [p.unsqueeze(0) if p.dim() == 0 else p for p in preds]
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return labels, preds


# Evaluation metrics
def calculate_optimal_threshold(y_true, y_proba):
    """Calculate the decision threshold."""

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr  # Youden's J = sensitivity + specificity - 1
    optimal_idx = np.nanargmax(j_scores)
    return thresholds[optimal_idx]


def performance_evaluation_proba(y_true, y_proba, optimal_threshold=None):
    """Compute classification metrics from positive-class probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Positive-class probabilities in [0, 1].
        optimal_threshold: Decision threshold; if None, the Youden-J optimal
            threshold from the ROC curve is used.

    Returns:
        Tuple (predictions, results): integer prediction array and a dict of
        accuracy, precision, recall, f1_score, specificity, auc, aupr, optimal_threshold, and confusion_matrix.
    """

    y_proba = np.asarray(y_proba, dtype=float)
    if optimal_threshold is None:
        optimal_threshold = calculate_optimal_threshold(y_true, y_proba)
    predictions = (y_proba >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_true, predictions)
    if conf_matrix.shape == (2, 2):
        tn, fp, _, _ = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = np.nan
    auc_score = roc_auc_score(y_true, y_proba)
    aupr_score = average_precision_score(y_true, y_proba)
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc': auc_score,
        'aupr': aupr_score,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': conf_matrix,
    }
    return predictions, results


def performance_evaluation(y_true, pred, optimal_threshold=None):
    """Compute classification metrics from logits.

    Args:
        y_true: Ground-truth binary labels.
        pred: Raw logits.
        optimal_threshold: Decision threshold; if None, chosen via Youden's J.

    Returns:
        Tuple (predictions, results).
    """

    y_proba = torch.sigmoid(torch.from_numpy(pred)).cpu().numpy()
    return performance_evaluation_proba(y_true, y_proba, optimal_threshold)


# Uncertainty / calibration
def compute_uncertainty(prob_arr):
    """Decompose ensemble predictive uncertainty into aleatoric and epistemic.

    Args:
        prob_arr: Array [n_members, n_samples] of per-member positive-class probabilities.

    Returns:
        Dict with mean_prob, aleatoric, epistemic, and total uncertainty, each a [n_samples] array.
    """

    prob_arr = np.asarray(prob_arr, dtype=float)
    if prob_arr.ndim != 2:
        raise ValueError(f"prob_arr must be [n_members, n_samples], got {prob_arr.shape}")
    mean_prob = prob_arr.mean(axis=0)
    aleatoric = (prob_arr * (1.0 - prob_arr)).mean(axis=0)
    epistemic = prob_arr.var(axis=0)
    total = aleatoric + epistemic
    return {
        "mean_prob": mean_prob,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total,
    }


def compute_calibration_errors(pred_probs, labels, n_bins=10):
    """Compute Expected (ECE) and Maximum (MCE) calibration error via uniform binning.

    Args:
        pred_probs: Predicted positive-class probabilities in [0, 1].
        labels: Ground-truth binary labels.
        n_bins: Number of equal-width probability bins.

    Returns:
        Tuple (ece, mce) of floats.
    """

    pred_probs = np.asarray(pred_probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (pred_probs > lo) & (pred_probs <= hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = labels[in_bin].mean()
            conf_in_bin = pred_probs[in_bin].mean()
            gap = abs(conf_in_bin - acc_in_bin)
            ece += gap * prop_in_bin
            mce = max(mce, gap)
    return ece, mce


    
# Collate / model / data loading
def build_unimol_encoder(device):
    """Instantiate a UniMol2 compound encoder using configuration hyperparameters."""

    from unimol_tools.models.unimolv2 import UniMolV2Model
    return UniMolV2Model(
        output_dim=config.UNIMOL_OUTPUT_DIM,
        model_size=config.UNIMOL_MODEL_SIZE,
        pooler_dropout=config.UNIMOL_POOLER_DROPOUT,
        remove_hs=config.UNIMOL_REMOVE_HS,
    ).to(device)

def build_collate_fn(unimol_model):
    """Build a DataLoader collate_fn bound to a UniMol2 model's batch collater."""

    def batch_collate_fn(samples):
        unimol2 = [[s[0]] for s in samples]
        unimol2 = unimol_model.batch_collate_fn(unimol2)
        esm1 = torch.stack([s[1] for s in samples])
        esm2 = torch.stack([s[2] for s in samples])
        auxiliary = torch.stack([s[3] for s in samples])
        valid_length = torch.stack([s[4] for s in samples])
        label = torch.FloatTensor([s[5] for s in samples])
        return unimol2, esm1, esm2, auxiliary, valid_length, label

    return batch_collate_fn


def build_model(device):
    """Build the PPIInhibitorModel (ICANPPII) architecture on the given device."""

    unimol_model = build_unimol_encoder(device)
    return PPIInhibitorModel(
        compound_encoder=unimol_model,
        dropout=config.DROPOUT,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
    ).to(device)


def load_model(device):
    """Build a training-ready PPIInhibitorModel with its loss and optimizer.

    Most of the UniMol2 encoder is frozen — only its last encoder layer (layer 11)
    and layer-norm parameters stay trainable — while parameters outside the
    encoder are trained.
    """
    
    model = build_model(device)

    # The last layer of UniMol2 is trainable.
    for name, param in model.named_parameters():
        if "compound_encoder.encoder.layers.11" in name:
            param.requires_grad = True
        elif "layer_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        if "compound_encoder" not in name:
            param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    return model, criterion, optimizer


def load_dataloader(split, collate_fn, device, which=("train", "valid", "test"), return_test_df=False):
    """Build DataLoaders for the requested dataset splits."""

    data_dir = config.split_dir(split)

    datasets = {}
    for name in which:
        datasets[name] = PPIInhibitorDataset(f"{data_dir}/{name}.csv", 
                                             config.ESM2_PATH, 
                                             config.PROT_PHY_PATH, 
                                             config.KG_PATH,
                                             config.COMP_PHY_PATH, 
                                             config.UNIMOL2_PATH, 
                                             device)

    loaders = {name: DataLoader(ds, 
                                batch_size=config.BATCH_SIZE,
                                shuffle=(name == "train"), 
                                drop_last=False,
                                collate_fn=collate_fn)
               for name, ds in datasets.items()
              }

    if return_test_df:
        return loaders, datasets["test"].get_df()
    return loaders
