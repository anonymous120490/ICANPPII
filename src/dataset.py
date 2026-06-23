import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset



class PPIInhibitorDataset(Dataset):
    """Dataset for PPI-inhibitor prediction."""

    def __init__(self, data_path, esm2_path, prot_phy_path, kg_path,
                 comp_phy_path, unimol2_path, device, maxlen=1022):
        """Load metadata and precomputed features and assemble valid samples.
 
        Args:
            data_path: CSV with uniprot_id1, uniprot_id2, can_smi (and optionally a label column).
            esm2_path: .pt dict mapping UniProt ID to its ESM-2 embedding.
            prot_phy_path: .pt dict of protein physicochemical descriptors.
            kg_path: .pt dict of protein knowledge-graph embeddings.
            comp_phy_path: .pt dict of compound physicochemical descriptors.
            unimol2_path: .pt dict mapping SMILES to UniMol2 inputs.
            device: Target torch device for the produced tensors.
            maxlen: Protein length for padding/truncation (default: 1022).
        """

        self.device = device
        self.maxlen = maxlen

        df = pd.read_csv(data_path)
        required_cols = ['uniprot_id1', 'uniprot_id2', 'can_smi']
        optional_cols = [c for c in ('ppi_label', 'label') if c in df.columns]
        df = df[required_cols + optional_cols]
        self.df = df
        self.has_label = 'label' in df.columns

        self.esm2_path = esm2_path
        self.prot_phy_path = prot_phy_path
        self.kg_path = kg_path
        self.comp_phy_path = comp_phy_path
        self.unimol2_path = unimol2_path

        self.process_compound()
        self.process_protein()

        self.df = self.df[self.df['can_smi'].isin(self.valid_smiles_list)].reset_index(drop=True)
        self.label_list = self.df['label'].tolist() if self.has_label else None


    def process_compound(self):
        """Load compound physicochemical features and UniMol2 inputs."""

        compound_phy = torch.load(self.comp_phy_path, weights_only=False)
        self.unimol2 = torch.load(self.unimol2_path, weights_only=False)
        self.compound_phy = {k: torch.as_tensor(v).float() for k, v in compound_phy.items()}
        self.valid_smiles_list = list(self.unimol2.keys())


    def process_protein(self):
        """Load ESM-2 embeddings, knowledge-graph embeddings, and descriptors."""

        esm = torch.load(self.esm2_path, weights_only=False)
        kg = torch.load(self.kg_path, weights_only=False)
        protein_phy = torch.load(self.prot_phy_path, weights_only=False)
        self.esm = {k: torch.as_tensor(v).float() for k, v in esm.items()}
        self.kg = {k: torch.as_tensor(v).float() for k, v in kg.items()}
        self.protein_phy = {k: torch.as_tensor(v).float() for k, v in protein_phy.items()}


    def pad_or_truncate(self, esm):
        """Pad or truncate a protein embedding to maxlen.

        Args:
            esm: Tensor of shape [L, D].

        Returns:
            Tuple of (Tensor of shape [maxlen, D], valid length before padding).
            Sequences longer than maxlen are truncated; shorter ones are zero-padded at the end.
        """

        valid_len = min(esm.size(0), self.maxlen)
        esm = esm[:valid_len, :]
        pad_len = self.maxlen - valid_len
        if pad_len > 0:
            esm = F.pad(esm, (0, 0, 0, pad_len))
        return esm, valid_len

    def __getitem__(self, idx):
        """Build the feature tuple of one sample.

        Args:
            idx: Row index into the filtered sample table.

        Returns:
            Tuple (unimol2, esm1, esm2, auxiliary, valid_length, label): the
            compound's UniMol2 input dict, the two padded protein embeddings,
            the concatenated physicochemical/KG/compound features, an IntTensor
            of the two pre-padding protein lengths, and the label (-1.0 when the
            dataset has no labels).
        """
        row = self.df.iloc[idx]
        smiles = row['can_smi']
        sample = self.unimol2[smiles]
        unimol2 = {
            'atom_feat': sample['atom_feat'],
            'atom_mask': sample['atom_mask'],
            'edge_feat': sample['edge_feat'],
            'shortest_path': sample['shortest_path'],
            'degree': sample['degree'],
            'pair_type': sample['pair_type'],
            'attn_bias': sample['attn_bias'],
            'src_tokens': sample['src_tokens'],
            'src_coord': sample['src_coord'],
        }

        uniprot_id1 = row['uniprot_id1']
        uniprot_id2 = row['uniprot_id2']

        esm1, len1 = self.pad_or_truncate(self.esm[uniprot_id1])
        esm2, len2 = self.pad_or_truncate(self.esm[uniprot_id2])

        auxiliary = torch.cat((self.protein_phy[uniprot_id1], self.kg[uniprot_id1],
                               self.protein_phy[uniprot_id2], self.kg[uniprot_id2],
                               self.compound_phy[smiles]), 0)

        valid_length = torch.IntTensor([len1, len2])
        label = self.label_list[idx] if self.has_label else -1.0
        return unimol2, esm1, esm2, auxiliary, valid_length, label


    def __len__(self):
        """Return the number of valid samples."""
        return len(self.df)


    def get_df(self):
        """Return the filtered sample table."""
        return self.df