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

def future_decode(
    outputs, 
    cfg: DictConfig,
    mmgpt: MultiModalityCausalLM,
    input_embeds_prompt: torch.Tensor,
    temperature: float = 1.0,
    inference = True,
    ):
    prefix_len = input_embeds_prompt.shape[1]
    
    #* amend past_key_values to only include prefix part
    past_key_values = ()
    for idx, layer_past_kv in enumerate(outputs.past_key_values):
        k_states, v_states = layer_past_kv
        k_states = k_states[:, :, :prefix_len, :]
        v_states = v_states[:, :, :prefix_len, :]
        past_key_values += ((k_states, v_states), )
     
    if cfg.yjk.target=="hidden_state":
        unpert = outputs.output_hidden_states.detach()
        
    elif cfg.yjk.target=="logit":
        unpert = outputs.output_logits.detach()
        
    unpert_cfg = cfg_decode(logit_cond = unpert[:, 0::2, :], logit_uncond = unpert[:, 1::2, :], scale = cfg.cfg_scale).permute(1, 0, 2) # image_tokens, bsz, vocab_size
    unpert_cfg = torch.softmax(unpert_cfg / temperature, dim=-1)
    epsilon = torch.nn.Parameter(torch.zeros_like(unpert), requires_grad=True) # image tokens, 2, 16384
    if cfg.yjk.optimizer=="adam":
        optimizer = torch.optim.Adam([epsilon], lr = cfg.yjk.stepsize)
    
    pbar = trange(cfg.yjk.ld_iters, desc="LD iterations")
    sigma = torch.linspace(5, 0.01, cfg.yjk.ld_iters)
    for idx in pbar:
        optimizer.zero_grad()
        #* Perturb logits
        # noise = torch.randn_like(unpert) * sigma[idx]
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
        next_token_hard = torch.argmax(probs, dim=-1)
        if inference:
            next_token_onehot = torch.nn.functional.one_hot(next_token_hard, num_classes=16384)
            curr_next_tokens = next_token_onehot - probs.detach() + probs
        else:
            curr_next_tokens = torch.nn.functional.gumbel_softmax(probs, tau=1, hard=True, dim=-1)
        
        curr_next_tokens = torch.cat([curr_next_tokens, curr_next_tokens], dim=1) # 2 * batch_size, img_tokens, 16384
        weights = mmgpt.gen_embed.weight  # [vocab_size, 8]
        img_embeds = torch.matmul(curr_next_tokens, weights) # 2 * batch_size, img_tokens, 2048
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
        loss_ppl = -torch.sum(probs_cfg * torch.log(probs_cfg + 1e-10)) / probs_cfg.size(0)
        loss_kl = torch.nn.functional.kl_div(probs_cfg.log(), unpert_cfg.detach(), reduction='batchmean')
        
        loss = loss_ppl + loss_kl
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f} Loss_PPL: {loss_ppl.item():.4f} Loss_KL: {loss_kl.item():.4f} Epsilon: {epsilon.abs().mean().item():.4f}")
        
    bsz = probs_cfg.shape[0]
    return torch.multinomial(probs_cfg.view(-1, probs_cfg.shape[-1]), num_samples=1).view(bsz, -1), torch.multinomial(unpert_cfg.view(-1, unpert_cfg.shape[-1]), num_samples=1).view(bsz, -1) # batch_size, img_tokens, vocab_size
    

    

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