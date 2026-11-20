import torch
from torch import nn
from typing import Optional

from slideflow import log
from slideflow.model.torch_utils import get_device

# -----------------------------------------------------------------------------


class Attention_IM_MIL(nn.Module):
    """Attention-based multiple instance learning model with intermediate fusioning.

    Implementation adapted from: https://github.com/KatherLab/marugoto
    """

    use_lens = True

    def __init__(
        self,
        n_feats: int,
        n_out: int,
        z_dim: int = 256,
        z_dim_clinical: int = 35,
        clinical_num_features: int = 0,
        *,
        dropout_p: float = 0.5,
        encoder: Optional[nn.Module] = None,
        clinical_encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        attention_gate: float = 0,
        temperature: float = 1.,
    ) -> None:
        """Create a new attention MIL model with intermediate fusioning of clinical data via Kronecker product.

        Args:
            n_feats (int): Total number of features per bag instance (image + clinical).
            n_out (int): Number of model outputs.
            z_dim (int): Dimensionality of image latent space. Defaults to 256.
            z_dim_clinical (int): Dimensionality of clinical latent space. Defaults to 35.
            clinical_num_features (int): Number of trailing clinical features concatenated to each instance.
                Must satisfy ``0 < clinical_num_features < n_feats``.
        """
        super().__init__()
        if clinical_num_features <= 0 or clinical_num_features >= n_feats:
            raise ValueError(
                f"clinical_num_features must satisfy 0 < clinical_num_features < n_feats ({n_feats}); "
                f"got {clinical_num_features}."
            )
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats - clinical_num_features, z_dim), nn.ReLU())
        self.attention = attention or Attention(z_dim)
        self.head = head or nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim * z_dim_clinical),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * z_dim_clinical, n_out),
        )
        self.clinical_encoder = clinical_encoder or nn.Sequential(nn.Linear(clinical_num_features, z_dim_clinical), nn.ReLU())
        self.clinical_num_features = clinical_num_features
        self._neg_inf = torch.tensor(-torch.inf)
        self.attention_gate = attention_gate
        self.temperature = temperature
        if temperature != 1.:
            log.debug("Using attention softmax temperature: {}".format(temperature))
        if attention_gate:
            log.debug("Using attention gate: {} percentile".format(attention_gate))

    def _split_features(self, bags: torch.Tensor):
        non_clinical = bags[..., :-self.clinical_num_features]
        clinical = bags[..., -self.clinical_num_features:]
        return non_clinical, clinical

    def forward(self, bags, lens, *, return_attention=False, uq=False, uq_softmax=True):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        non_clinical_features, clinical_features = self._split_features(bags)
        embeddings = self.encoder(non_clinical_features)
        masked_attention_scores = self._masked_attention_scores(embeddings, lens, apply_softmax=False)

        if self.attention_gate and bags.shape[1] > 1:
            attention_threshold = torch.quantile(masked_attention_scores, q=self.attention_gate, dim=1, keepdim=True)
            high_attention_mask = (masked_attention_scores > attention_threshold)[:, :]
            masked_attention_scores = torch.where(
                high_attention_mask, masked_attention_scores, torch.full_like(masked_attention_scores, self._neg_inf)
            )

        softmax_attention_scores = torch.softmax(masked_attention_scores / self.temperature, dim=1)
        weighted_embeddings = softmax_attention_scores * embeddings
        weighted_embeddings_sum = weighted_embeddings.sum(-2)

        clinical_features_mean = clinical_features.mean(-2)
        clinical_embeddings = self.clinical_encoder(clinical_features_mean)

        # Kronecker product across squeezed latent feature vectors: (B, z_dim) x (B, z_dim_clinical) -> (B, z_dim * z_dim_clinical)
        final_embeddings = torch.einsum(
            'bi,bj->bij',
            weighted_embeddings_sum.squeeze(1),
            clinical_embeddings.squeeze(1),
        ).reshape(weighted_embeddings_sum.shape[0], -1)

        if uq:
            flat = self.head[0](final_embeddings)
            norm = self.head[1](flat)
            expanded = norm.unsqueeze(1).expand(-1, 30, -1)

            _prior_status = self.training
            self.head[2].train()
            post_dropout = self.head[2](expanded)
            expanded_preds = self.head[3](post_dropout)
            if uq_softmax:
                expanded_preds = torch.softmax(expanded_preds, dim=2)
            self.train(_prior_status)

            pred_stds = torch.std(expanded_preds, dim=1)
            pred_means = expanded_preds.mean(axis=1)
            scores = (pred_means, pred_stds)
        else:
            scores = self.head(final_embeddings)

        if return_attention and bags.shape[1] > 1:
            return scores, softmax_attention_scores
        elif return_attention:
            return scores, masked_attention_scores
        else:
            return scores

    def calculate_attention(self, bags, lens, *, apply_softmax=None):
        """Calculate attention scores for all bags."""
        if apply_softmax is None and bags.shape[1] > 1:
            apply_softmax = True
        non_clinical_features, _ = self._split_features(bags)
        embeddings = self.encoder(non_clinical_features)
        return self._masked_attention_scores(
            embeddings, lens, apply_softmax=apply_softmax
        )

    def get_last_layer_activations(self, bags, lens):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]
        non_clinical_features, _ = self._split_features(bags)
        embeddings = self.encoder(non_clinical_features)
        masked_attention_scores = self._masked_attention_scores(embeddings, lens)
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)
        return weighted_embedding_sums

    def _masked_attention_scores(self, embeddings, lens, *, apply_softmax=True):
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)
        idx = torch.arange(bag_size, device=attention_scores.device).unsqueeze(0).expand(bs, -1)
        attention_mask = (idx < lens.to(attention_scores.device).unsqueeze(-1)).unsqueeze(-1)
        masked_attention = torch.where(
            attention_mask, attention_scores, torch.full_like(attention_scores, self._neg_inf)
        )
        if apply_softmax:
            return torch.softmax(masked_attention, dim=1)
        else:
            return masked_attention

    def relocate(self):
        """Move model to GPU. Required for FastAI compatibility."""
        device = get_device()
        self.to(device)
        self._neg_inf = self._neg_inf.to(device)

    def plot(*args, **kwargs):
        pass


# -----------------------------------------------------------------------------


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    if n_latent == 0:
        return nn.Linear(n_in, 1)
    else:
        n_latent = n_latent or (n_in + 1) // 2
        return nn.Sequential(
            nn.Linear(n_in, n_latent),
            nn.Tanh(),
            nn.Linear(n_latent, 1),
        )
