import torch
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange
from omegaconf import DictConfig
from einops import rearrange, repeat
from janus.models import MultiModalityCausalLM, VLChatProcessor



def cfg_decode(
    logit_cond,
    logit_uncond,
    scale,
    **kwargs
    ):
    logits = scale * logit_cond + (1 - scale) * logit_uncond
    return logits

def vanilla_decode(
    logit_cond,
    **kwargs
    ):
    return logit_cond


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


def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    softmax = torch.softmax(y / tau, dim=-1)
    
    if hard:
        # Convert softmax probabilities to one-hot for the forward pass
        idx = softmax.argmax(dim=-1, keepdim=True)
        hard_one_hot = torch.zeros_like(softmax).scatter_(-1, idx, 1.0)
        return (hard_one_hot - softmax).detach() + softmax, idx  # Straight-through gradient
    else:
        return softmax

def soft_argmax(soft_sample):
    """
    Compute a soft version of argmax that outputs differentiable indices.
    Args:
        soft_sample: Tensor of shape (..., num_classes), soft probabilities over classes.
    Returns:
        Tensor of shape (...), differentiable indices.
    """
    num_classes = soft_sample.size(-1)
    indices = torch.arange(num_classes, device=soft_sample.device)
    return torch.sum(soft_sample * indices, dim=-1)

def soft_nll(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)


def future_decode(
    outputs, 
    cfg: DictConfig,
    mmgpt: MultiModalityCausalLM,
    input_embeds_prompt: torch.Tensor,
    temperature: float = 1.0,
    ):
    prefix_len = input_embeds_prompt.shape[1]
    
    #* amend past_key_values to only include prefix part
    past_key_values = ()
    for idx, layer_past_kv in enumerate(outputs.past_key_values):
        k_states, v_states = layer_past_kv
        k_states = k_states[:, :, :prefix_len, :] # 2, 
        v_states = v_states[:, :, :prefix_len, :]
        past_key_values += ((k_states, v_states), )
     
    if cfg.yjk.target=="hidden_state":
        unpert = outputs.output_hidden_states.detach()
        all_input_ids = torch.arange(0, 16384, dtype=torch.int, device=unpert.device)
        hidden_state_dict = mmgpt.language_model.model.embed_tokens(all_input_ids) # vocab_size, 2048 
        def map_hidden_state(hidden_states, hidden_state_dict):
            """
            Maps each vector in hidden_states to the closest vector in hidden_state_dict.

            Args:
                hidden_states (torch.Tensor): Tensor of shape (N, 2048).
                hidden_state_dict (torch.Tensor): Tensor of shape (M, 2048).

            Returns:
                torch.Tensor: Mapped hidden states of shape (N, 2048).
            """
            distances = torch.cdist(hidden_states, hidden_state_dict)  # Shape: (N, M)
            closest_indices = torch.argmin(distances, dim=-1)  # Shape: (N,)
            mapped_hidden_states = hidden_state_dict[closest_indices]
            return mapped_hidden_states
        unpert = map_hidden_state(unpert, hidden_state_dict) #* CUDA OOM error
        
    elif cfg.yjk.target=="logit":
        unpert = outputs.output_logits.detach()
        
    epsilon = torch.nn.Parameter(torch.zeros_like(unpert), requires_grad=True) # image tokens, 2, 16384
    if cfg.yjk.optimizer=="adam":
        optimizer = torch.optim.Adam([epsilon], lr = cfg.yjk.stepsize)
    
    pbar = trange(cfg.yjk.ld_iters, desc="LD iterations")
    sigma = torch.linspace(5, 0.01, cfg.yjk.ld_iters)
    for idx in pbar:
        optimizer.zero_grad()
        #* Perturb logits
        noise = torch.randn_like(unpert) * sigma[idx]
        pert_logits = unpert + epsilon # + noise  # perturbed
        
        if pert_logits.shape[-1] != 16384:
            pert_logits = mmgpt.gen_head(pert_logits) # n_img_tokens, bsz * 2, vocab_size
        
        #* Apply CFG
        uncond_logits, cond_logits = pert_logits[:, 0::2, :], pert_logits[:, 1::2, :]
        logits = cfg_decode(logit_cond = cond_logits, logit_uncond = uncond_logits, scale = cfg.cfg_scale) # image_tokens, bsz, vocab_size
        probs = torch.softmax(logits / temperature, dim=-1) # image_tokens, bsz, vocab_size
        
        #* Sample tokens top-k on perturbed CFG logits
        # next_token = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
        # next_token = torch.cat([next_token, next_token], dim=1)  # image_tokens, bsz * 2 
        # emb = mmgpt.gen_embed(next_token) # image_tokens, bsz * 2, 8
        # img_embeds = mmgpt.gen_aligner(emb).permute(1, 0, 2) # 2 * batch_size, img_tokens, 2048
        
        #* Get output logits from sampled tokens
        next_token_hard = probs.argmax(dim=-1, keepdim=True) # use gumbel_softmax to relax greedy
        next_token_hard = (next_token_hard - probs).detach() + probs
        next_token_hard = torch.cat([next_token_hard, next_token_hard], dim=1) # 2 * batch_size, img_tokens, 16384
        weights = mmgpt.gen_embed.weight  # [vocab_size, 8]
        img_embeds = torch.matmul(next_token_hard, weights) # 2 * batch_size, img_tokens, 2048
        img_embeds = mmgpt.gen_aligner(img_embeds).permute(1, 0, 2) # 2 * batch_size, img_tokens, 2048
        
        args = {
            "inputs_embeds": img_embeds, # bsz, img_tokens, 2048
            "attention_mask": None,
            "use_cache": True,
            "past_key_values": past_key_values, # 2 * batch_size, n_heads, prefix_len, 2048
            "output_hidden_states": False,
        }
        outputs = mmgpt.language_model.model(
            **args
        )
        hidden_states = outputs.last_hidden_state # 2 * batch_size, img_tokens, 2048
        logits = mmgpt.gen_head(hidden_states) # 2 * batch_size, img_tokens, 16384
        logits_cfg = cfg_decode(logit_cond = logits[0::2, :], logit_uncond = logits[1::2, :], scale = cfg.cfg_scale) # 2 * batch_size, img_tokens, 16384
        
        #! Fluency Constraint of BOLT = same as perplexity loss
        probs_cfg = torch.softmax(logits_cfg / temperature, dim=-1)  # batch_size, img_tokens, vocab_size
        loss = -torch.sum(probs_cfg * torch.log(probs_cfg + 1e-10)) / probs_cfg.size(0)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f} Epsilon: {epsilon.abs().mean().item():.4f}")
    
    return probs_cfg.argmax(dim=-1), torch.multinomial(probs, num_samples=1) # batch_size, img_tokens
    

    

def rollout(
    mmgpt: MultiModalityCausalLM,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    rollout_len: int,
    cfg: DictConfig,
    temperature: float = 1.0,
    ):
    uncond_p_stack = torch.zeros((batch_size, rollout_len, 16384)).cuda()
    cond_p_stack = torch.zeros_like(uncond_p_stack).cuda()
    for k in range(rollout_len):
        outputs = mmgpt.language_model.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            use_cache = True,
            past_key_values = outputs.past_key_values if k else None,
            output_hidden_states = False,
        )
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        uncond_logits, cond_logits = logits[0::2, :], logits[1::2, :]
        uncond_p, cond_p = torch.softmax(uncond_logits, dim=-1), torch.softmax(cond_logits, dim=-1)
        uncond_p_stack[:, k, :], cond_p_stack[:, k, :] = uncond_p, cond_p
        next_token_uncond, next_token_cond = torch.multinomial(uncond_p, num_samples=1), torch.multinomial(cond_p, num_samples=1)
        inputs_embeds = mmgpt.prepare_gen_img_embeds(
            torch.cat([next_token_uncond, next_token_cond], dim=1).view(-1)
        ).unsqueeze(
            dim=1
                ) # 8 [uncond, cond], 1, 2048
    uncond_p, cond_p = torch.softmax(uncond_p_stack.mean(dim=1), dim=-1), torch.softmax(cond_p_stack.mean(dim=1), dim=-1)
    logits = cfg_decode(logit_cond = cond_p, logit_uncond = uncond_p, scale = cfg.cfg_scale, **kwargs)
    probs = torch.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)