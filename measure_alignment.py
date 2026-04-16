"""Feature preparation helpers shared by alignment scripts."""
import torch


def _remove_outliers(feats, q, exact=False):
    if q == 1:
        return feats
    if exact:
        q_val = feats.reshape(-1).abs().sort().values[int(q * feats.numel())]
    else:
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()
    return feats.clamp(-q_val, q_val)


def prepare_features(feats, q=0.95, exact=False):
    if isinstance(feats, torch.Tensor):
        return _remove_outliers(feats.float(), q=q, exact=exact)
    if isinstance(feats, list):
        return [_remove_outliers(f.float(), q=q, exact=exact) for f in feats]
    raise ValueError(f"Unsupported input type for prepare_features: {type(feats)}")
