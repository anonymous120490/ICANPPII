import torch
from torch import nn


def make_key_padding_mask(lengths, max_len):
    """Build a key-padding mask.

    Args:
        lengths: LongTensor [B], unpadded length of each sequence.
        max_len: int, padded sequence length.

    Returns:
        BoolTensor [B, max_len] (True = pad).
    """
    device = lengths.device
    range_row = torch.arange(max_len, device=device).unsqueeze(0)  # [1, L]
    mask = range_row >= lengths.unsqueeze(1)                       # [B, L]
    return mask  # True = pad


def masked_mean(x, key_padding_mask):
    """Mean over the sequence axis, ignoring pad positions.

    Args:
        x: [B, L, D]
        key_padding_mask: [B, L] (True = pad), or None for a plain mean.

    Returns:
        [B, D]
    """
    if key_padding_mask is None:
        return x.mean(dim=1)
    mask = (~key_padding_mask).float().unsqueeze(-1)  # [B, L, 1]
    x_sum = (x * mask).sum(dim=1)                     # [B, D]
    denom = mask.sum(dim=1).clamp_min(1e-6)           # [B, 1]
    return x_sum / denom


def pad_3d_sequences(seq_list, pad_value=0.0):
    """Right-pad variable-length [Ni, D] tensors into one [B, Nmax, D] batch.

    Args:
        seq_list: List of [Ni, D] tensors sharing the same D and device.
        pad_value: Fill value for padded positions.

    Returns:
        Tuple of (padded tensor [B, Nmax, D], LongTensor [B] of original lengths).
    """
    device = seq_list[0].device
    B = len(seq_list)
    D = seq_list[0].size(-1)
    lengths = torch.tensor([t.size(0) for t in seq_list], device=device, dtype=torch.long)
    Nmax = int(lengths.max().item())

    padded = seq_list[0].new_full((B, Nmax, D), fill_value=pad_value)
    for i, t in enumerate(seq_list):
        Ni = t.size(0)
        padded[i, :Ni] = t
    return padded, lengths


class TransformerCrossBlock(nn.Module):
    """Cross-attention block"""

    def __init__(self, d_model=64, n_heads=1, d_ff=256, dropout=0.1):
        """Build the cross-attention block's sublayers.

        Args:
            d_model: Model dimension of all transformer sublayers.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout rate.
        """

        super().__init__()
        self.ln_q1 = nn.LayerNorm(d_model)
        self.ln_kv1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=n_heads,
                                          dropout=dropout,
                                          batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.ln_q2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, Q, K, V, key_padding_mask=None, attn_mask=None):
        """Let query Q attend to key/value K, V with Pre-LN and residuals.

        Args:
            Q: Query [B, Lq, D].
            K, V: Key/value [B, Lk, D].
            key_padding_mask: [B, Lk] over K/V (True = pad).
            attn_mask: Attention mask.

        Returns:
            Updated query [B, Lq, D].
        """

        Qn = self.ln_q1(Q)
        Kn = self.ln_kv1(K)
        Vn = self.ln_kv1(V)
        attn_out, _ = self.attn(Qn, Kn, Vn,
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask,
                                need_weights=False)
        Q = Q + self.dropout(attn_out)
        Qn2 = self.ln_q2(Q)
        Q = Q + self.ffn(Qn2)
        return Q


class PPIInhibitorModel(nn.Module):
    """ICANPPII main model.

    Predicts PPI inhibition from a compound and two proteins via cross-attention
    networks, fusing six DTI/PPI relational features with auxiliary physicochemical/KG features.
    """

    def __init__(self, compound_encoder, dropout=0.2, d_model=64, n_heads=2, d_ff=256):
        """Assemble the encoders, cross-attention blocks, and predictor head.

        Args:
            compound_encoder: Pretrained UniMol2 encoder.
            dropout: Dropout rate.
            d_model: Model dimension of all transformer sublayers.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
        """

        super().__init__()

        self.compound_encoder = compound_encoder
        self.protein_fc = nn.Linear(1280, d_model)
        self.compound_fc = nn.Linear(768, d_model)

        self.cross_dti1 = TransformerCrossBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.cross_dti2 = TransformerCrossBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.cross_dti3 = TransformerCrossBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.cross_dti4 = TransformerCrossBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.cross_ppi1 = TransformerCrossBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.cross_ppi2 = TransformerCrossBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)

        self.contextual_fc = nn.Sequential(
            nn.LayerNorm(6 * d_model),
            nn.Linear(6 * d_model, 128),
            nn.GELU()
        )
        self.auxiliary_fc = nn.Linear(86, 64)

        self.fuse_norm = nn.LayerNorm(128 + 64)

        self.fc1 = nn.Linear(128 + 64, 128)  # contextual(128) + aux(64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, unimol2, esm1, esm2, auxiliary, valid_length):
        """Predict a PPI-inhibition logit.

        Args:
            unimol2: Dict of UniMol2 batch inputs.
            esm1: Protein1 token embeddings [B, L1, 1280].
            esm2: Protein2 token embeddings [B, L2, 1280].
            auxiliary: Physicochemical and KG features [B, 86].
            valid_length: unpadded lengths of protein1/2 [B, 2].

        Returns:
            Logit tensor [B, 1].
        """

        # 1) Encode compound and proteins.
        output = self.compound_encoder(**unimol2, return_repr=True, return_atomic_reprs=True)
        compound_list = output['atomic_reprs']
        compound, num_atom = pad_3d_sequences(compound_list)
        compound = self.compound_fc(compound)  # [B, Na, d_model]
        esm1 = self.protein_fc(esm1)           # [B, L1, d_model]
        esm2 = self.protein_fc(esm2)           # [B, L2, d_model]

        # 2) Key-padding masks.
        L1, L2 = esm1.size(1), esm2.size(1)
        num_resi1, num_resi2 = valid_length[:, 0], valid_length[:, 1]
        kpm_comp = make_key_padding_mask(num_atom, compound.size(1))
        kpm_p1 = make_key_padding_mask(num_resi1, L1)
        kpm_p2 = make_key_padding_mask(num_resi2, L2)

        # 3) Cross-attention.
        # Compound <-> Protein1
        comp_q_from_p1 = self.cross_dti1(Q=compound, K=esm1, V=esm1, key_padding_mask=kpm_p1, attn_mask=None)
        p1_q_from_comp = self.cross_dti2(Q=esm1, K=compound, V=compound, key_padding_mask=kpm_comp, attn_mask=None)

        # Compound <-> Protein2
        comp_q_from_p2 = self.cross_dti3(Q=compound, K=esm2, V=esm2, key_padding_mask=kpm_p2, attn_mask=None)
        p2_q_from_comp = self.cross_dti4(Q=esm2, K=compound, V=compound, key_padding_mask=kpm_comp, attn_mask=None)

        # Protein1 <-> Protein2
        p1_q_from_p2 = self.cross_ppi1(Q=esm1, K=esm2, V=esm2, key_padding_mask=kpm_p2, attn_mask=None)
        p2_q_from_p1 = self.cross_ppi2(Q=esm2, K=esm1, V=esm1, key_padding_mask=kpm_p1, attn_mask=None)

        # 4) Masked pooling.
        dti1_vec = masked_mean(comp_q_from_p1, kpm_comp)  # [B, d_model]
        dti2_vec = masked_mean(comp_q_from_p2, kpm_comp)  # [B, d_model]
        dti3_vec = masked_mean(p1_q_from_comp, kpm_p1)    # [B, d_model]
        dti4_vec = masked_mean(p2_q_from_comp, kpm_p2)    # [B, d_model]
        ppi1_vec = masked_mean(p1_q_from_p2, kpm_p1)      # [B, d_model]
        ppi2_vec = masked_mean(p2_q_from_p1, kpm_p2)      # [B, d_model]

        # 5) Contextual representation fusion.
        contextual = torch.cat([dti1_vec, dti2_vec, dti3_vec, dti4_vec, ppi1_vec, ppi2_vec], dim=-1)  # [B, 6*d_model]
        contextual = self.contextual_fc(contextual) 

        # 6) Auxiliary features.
        auxiliary = self.auxiliary_fc(auxiliary)  # [B, 64]

        # 7) Predictor.
        x = torch.cat([contextual, auxiliary], dim=-1)  # [B, 192]
        x = self.fuse_norm(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.out(x)  # [B, 1]
        return x