import os

# Path
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output")
FEATURES_DIR = os.path.join(DATA_ROOT, "features")

ESM2_PATH = os.path.join(FEATURES_DIR, "esm_650M_embedding.pt")
PROT_PHY_PATH = os.path.join(FEATURES_DIR, "protein_phy.pt")
KG_PATH = os.path.join(FEATURES_DIR, "kg_embedding.pt")
COMP_PHY_PATH = os.path.join(FEATURES_DIR, "compound_phy.pt")
UNIMOL2_PATH = os.path.join(FEATURES_DIR, "unimol2_input_features.pt")


# Experiment settings
SEED = 1
SPLITS = ["random", "scaffold", "sequence", "knowledge"]
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 30
MAXLEN = 1022


# Cross-attention configuration
D_MODEL = 64
N_HEADS = 2
D_FF = 256
DROPOUT = 0.2


# UniMol2 compound encoder.
UNIMOL_MODEL_SIZE = "84m"
UNIMOL_OUTPUT_DIM = 2
UNIMOL_POOLER_DROPOUT = 0.1
UNIMOL_REMOVE_HS = False


# Checkpoint model configuration
MODEL_NAME = "ICANPPII"


# Path helpers
def split_dir(split):
    """Returns the base dataset directory for a given split."""
    return os.path.join(DATA_ROOT, split)


def output_dir(split):
    """Returns the output directory for a given split."""
    return os.path.join(OUTPUT_ROOT, split)


def model_path(split, seed=SEED):
    """Returns the file path for the saved model."""
    return os.path.join(output_dir(split), f"{MODEL_NAME}_seed{seed}.pt")


def results_path(split, seed=SEED):
    """Returns the file path for the evaluation results CSV."""
    return os.path.join(output_dir(split), f"{MODEL_NAME}_seed{seed}_eval.csv")


def prediction_path(split, seed=SEED):
    """Returns the file path for the single-seed prediction CSV."""
    return os.path.join(output_dir(split), f"prediction_seed{seed}.csv")


def ensemble_prediction_path(split):
    """Returns the file path for the ensemble prediction CSV."""
    return os.path.join(output_dir(split), "prediction_ensemble.csv")