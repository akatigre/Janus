
import math
import torch
import numpy as np
import torch
from torch.nn import functional as F

def cfg_decode(
    logit_cond,
    logit_uncond,
    scale,
    **kwargs
    ):
    logits = (1 + scale) * logit_cond - scale * logit_uncond
    return logits

def vanilla_decode(
    logit_cond,
    **kwargs
    ):
    return logit_cond

def myopic_decode(
    logit_cond,
    logit_uncond,
    scale,
    forward_func,
    **kwargs
    ):
    logits = logit_uncond
    return logits

def adaptive_decode(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, ada: float = 0.01, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    assert ada > 0.0005 and ada < 0.01, f"adaptive threshold should be between 0.0005 and 0.01, got {ada}"
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    prob = sorted_logits.softmax(dim=-1)
    cumulative_probs = prob.cumsum(dim=-1)

    vocab_size = cumulative_probs.shape[1]
    up_bound = -np.log(1.0 / vocab_size)
    position = torch.arange(1, vocab_size + 1).repeat(cumulative_probs.shape[0], 1).to(cumulative_probs.device)

    A = prob * torch.log(prob * (vocab_size - position) / (1.0 - cumulative_probs))
    B = (1 - cumulative_probs) / (vocab_size - position)
    C = (1 - cumulative_probs + prob) / (vocab_size + 1 - position)
    delta_conf = (A + (1 - cumulative_probs + prob) * torch.log(B / C)) / up_bound
    delta_conf[torch.isnan(delta_conf)] = 0

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = delta_conf <= ada

    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed