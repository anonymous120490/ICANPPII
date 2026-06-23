# Interpretable Cross-Attention Network for Predicting Protein-Protein Interaction Inhibitors

## Abstract
Protein-protein interactions (PPIs) are attractive yet challenging therapeutic targets, as their large and featureless interfaces are difficult to engage with small molecules. Critically, the vast majority of disease-associated PPIs have no known inhibitor, placing them beyond the reach of conventional data-driven models. Here, this study presents ICANPPII, an interpretable and uncertainty-aware deep learning model that predicts PPI inhibitors and extends candidate discovery to these previously uncharacterized targets. The model integrates pretrained embeddings of protein sequences and compound structures with knowledge graph embeddings that encode biological context and key physicochemical properties. A cross-attention mechanism learns meaningful intermolecular relationships while providing interpretable insights into interaction patterns between protein interfaces and small molecules. To improve reliability, Deep Ensembles are employed to quantify predictive uncertainty, mitigating overconfident predictions and reducing false positives in high-throughput screening. Through comprehensive validation, ICANPPII outperforms existing methods, particularly in the most challenging unseen-protein scenarios that mirror inhibitor discovery for novel targets. Applied to large-scale virtual screening, it identifies novel candidate inhibitors of the KRAS-SOS1 interaction.  This work offers a powerful and practical computational tool for discovering novel therapeutic candidates targeting protein-protein interactions. 

---

## Usage

## 1. Install dependencies

```bash
conda create -n ICANPPII python=3.11
conda activate ICANPPII

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install rdkit qeppi numpy pandas scipy scikit-learn matplotlib seaborn tmap statannotations jupyter ipykernel tqdm fsspec fair-esm unimol-tools biopython pfeature
conda install conda-forge::faerun mkl=2024.0
```

---

## 2. Pretrained weights

You can download the ICANPPII model weights.

```bash
pip install gdown
mkdir -p model_weights
```

| Split |  Command |
|-----------|------------------------|
| random | `gdown "https://drive.google.com/file/d/1AFm-dRHZnpg_sgJwj0hk8NmvJOK6r8-W/view?usp=sharing"    -O model_weights/random.tar.gz` |
| scaffold | `gdown "https://drive.google.com/file/d/182dWZ7u7NR6FKfNj_bcELdZ8W_5429MR/view?usp=sharing"  -O model_weights/scaffold.tar.gz` |
| sequence | `gdown "https://drive.google.com/file/d/17wj2j_lw-pLeqzV9kGrgU2hmIINqIlt3/view?usp=sharing"  -O model_weights/sequence.tar.gz` |
| knowledge | `gdown "https://drive.google.com/file/d/1njm9Aa1iNKyJEUfr15WUpup0LplXHFka/view?usp=sharing" -O model_weights/knowledge.tar.gz` |


To unpack the downloaded tar.gz files,
```bash
cd model_weights && for f in *.tar.gz; do tar -xzf "$f"; done
```

---

## 3. Input format

Inputs should be CSV files with below columns depending on the task. 

| Task | Required columns |
|------|------------------|
| Training | `uniprot_id1`, `uniprot_id2`, `seq1`, `seq2`, `can_smi`, `label` |
| Inference | `uniprot_id1`, `uniprot_id2`, `seq1`, `seq2`, `can_smi` |

- `seq1`, `seq2`: amino acid sequences of the two proteins.
- `can_smi`: canonical SMILES of the compound.
- `label`: inhibition label (0/1) — required for training/evaluation only.


---

## 4. Training

To use your own dataset or model, before training:

1. Format your CSV to match Section 3.
2. Update `config.py` for your layout, e.g.:
   - `DATA_ROOT` — dataset root directory (default `data`)
   - `OUTPUT_ROOT` — checkpoint/results/predictions root (default `output`)
   - `MODEL_NAME` — checkpoint base name (default `ICANPPII`)


### 4.1 Featurization


```bash
# Proteins
python src/preprocess/compute_esm2.py --input data/random/test.csv --output data/features/esm_650M_embedding.pt --device cuda
python src/preprocess/compute_protein_phys.py --input data/random/test.csv --output data/features/protein_phy.pt
python src/preprocess/compute_kg_embedding.py --input data/random/test.csv --output data/features/kg_embedding.pt
```
> `compute_kg_embedding.py` maps a precomputed knowledge graph embedding (queried by UniProt ID) to your proteins.  
> It fills missing proteins with zeros.


```bash
# Compounds
python src/preprocess/compute_unimol2_input.py --input data/random/test.csv --output data/features/unimol2_input_features.pt --device cuda
python src/preprocess/compute_compound_phys.py --input data/random/test.csv --output data/features/compound_phy.pt
```



### 4.2 Train

```bash
python train.py --splits random --seeds 1 2 3 4 5 --epochs 30 --device cuda
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--splits` | all splits | Data split strategy (`random`, `scaffold`, `sequence`, `knowledge`). |
| `--seeds` | `1 2 3 4 5` | Seeds to train. |
| `--epochs` | `config.NUM_EPOCHS` | Training epochs. |
| `--device` | `cuda` if available, else `cpu` | Compute device. |





### 4.3 Evaluate

```bash
python evaluate.py --splits random --seeds 1 2 3 4 5 --device cuda
```

---

## 5. Inference

Run inference on new data with trained checkpoints.

```bash
python predict.py --input data/random/test.csv --output output/prediction.csv --checkpoints model_weights/ICANPPII_seed1.pt model_weights/ICANPPII_seed2.pt --device cuda
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Input CSV (`uniprot_id1`, `uniprot_id2`, `seq1`, `seq2`, `can_smi`). |
| `--output` | (required) | Output CSV file path. |
| `--checkpoints` | (required) | One or more checkpoint `.pt` paths; multiple is treated as ensemble. |
| `--device` | `cuda` if available, else `cpu` | Compute device. |

---

## Repository structure

```
.
├── config.py              # hyperparameters, paths
├── train.py               # training 
├── evaluate.py            # evaluation
├── predict.py             # inference (no label)
├── README.md
├── data/
│   ├── random/            # train.csv, valid.csv, test.csv
│   ├── scaffold/          # train.csv, valid.csv, test.csv
│   ├── sequence/          # train.csv, valid.csv, test.csv
│   ├── knowledge/         # train.csv, valid.csv, test.csv
│   └── features/          # featurization outputs (*.pt)
├── output/                # checkpoints, eval CSVs, prediction CSVs per split
└── src/
    ├── model.py           # PPIInhibitorModel
    ├── dataset.py         # PPIInhibitorDataset
    ├── utils.py           # training, evaluation, and data utilities
    └── preprocess/        # compute_*.py featurization scripts
```

---


## License
This project is licensed under the CC BY-NC-SA 4.0 License.
