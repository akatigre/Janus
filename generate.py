import torch
from omegaconf import DictConfig
from janus.models import MultiModalityCausalLM
from decode import cfg_decode
from tqdm import trange

def generate_t2i(
        mmgpt: MultiModalityCausalLM,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int = 16,
        image_token_num_per_image: int = 576,
        temperature: float = 1.0,
        cfg: DictConfig = None,
        do_langevin_dynamics: bool = True,
        epsilons: torch.Tensor = None,
        **kwargs,
    ): 
    output_input_embeds = torch.zeros((2 * batch_size, image_token_num_per_image, 2048)).cuda()
    generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()
    output_soft_onehots = torch.zeros((batch_size, image_token_num_per_image, 16384)).cuda()
    inputs_embeds = prompt_embeds
    prefix_len = prompt_embeds.shape[1]
    for i in trange(image_token_num_per_image):    
        
        args = {
            "inputs_embeds": inputs_embeds, # 8, 13, 2048
            "attention_mask": attention_mask, # 8, 13 = padding mask to indicate padding tokens
            "use_cache": True,
            "past_key_values": outputs.past_key_values if i != 0 else None,
            "output_hidden_states": False,
        }
        with torch.inference_mode():
            outputs = mmgpt.language_model.model(
                **args
            )
        
        if not i:
            hidden_states = outputs.last_hidden_state[:, -1, :].unsqueeze(dim=1) # for prefix, remove
        else:
            hidden_states = outputs.last_hidden_state
            
        _, _, next_tokens, _, _, cfg_logits, *_ = soft_forward(
            hidden_states=hidden_states.clone().detach().requires_grad_(True), 
            mmgpt=mmgpt, 
            soft = True if do_langevin_dynamics else False,
            token_idx = i,
            cfg = cfg,
            use_hidden_state_bias = cfg.yjk.use_hidden_state_bias,
            total_img_tokens = image_token_num_per_image,
            epsilon = epsilons[:, i, :].unsqueeze(dim=1) if do_langevin_dynamics else None
            )
        generated_tokens[:, i:i+1] = next_tokens
        cfg_logits[:, i:i+1] = cfg_logits
    return generated_tokens, cfg_logits
    
def soft_forward(hidden_states, mmgpt: MultiModalityCausalLM, epsilon : torch.Tensor, token_idx: int, soft: bool = True, temperature: float = 1.0, cfg: DictConfig = None, use_hidden_state_bias: bool = False, total_img_tokens: int = 576):
    assert hidden_states.dim() == 3, "hidden_states should be of shape (2 * batch_size, seq_len, 2048)"
    bsz = hidden_states.shape[0] // 2
    uncond_hidden_states = hidden_states[0::2]
    cond_hidden_states = hidden_states[1::2]
    last_logits = mmgpt.gen_head(hidden_states) # 2 * batch_size, seq_len, vocab_size (16384)
    uncond_logits, cond_logits = last_logits[0::2, :], last_logits[1::2, :]
    cfg_logits = cfg_decode(logit_cond = cond_logits, logit_uncond = uncond_logits, scale = cfg.cfg_scale)
    
    if epsilon is not None:
        weight = 1 - token_idx / total_img_tokens
        
        if not use_hidden_state_bias:
            cfg_logits = cfg_logits + weight * epsilon
        else:
            cfg_logits = cfg_logits + weight * mmgpt.gen_head(epsilon)
    
    # next_token_hard = torch.argmax(logits.view(-1, 16384), dim=-1).view(bsz, -1)
    
    cfg_probs = torch.softmax(cfg_logits / temperature, dim=-1)
    next_token_hard = torch.multinomial(cfg_probs.view(-1, 16384), num_samples=1).view(bsz, -1) # batch_size, seq_len
    if soft: # Straight Through Estimator
        next_token_onehot = torch.nn.functional.one_hot(next_token_hard, num_classes=16384) # batch_size, seq_len, 16384
        next_token_onehot = next_token_onehot - cfg_probs.detach() + cfg_probs
        weights = mmgpt.gen_embed.weight # mmgpt tensors are inference mode, so clone into regular tensor that records gradient
        img_embeds = torch.matmul(torch.cat([next_token_onehot, next_token_onehot], dim=0), weights) # 2 * batch_size, seq_len, 2048
        img_embeds = mmgpt.gen_aligner(img_embeds) # 2 * batch_size, seq_len, 2048
    else:
        next_token = torch.cat([next_token_hard, next_token_hard], dim=0)  # 2 * batch_size, seq_len
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token) # 2 * batch_size, seq_len, 2048
        next_token_onehot = None
    return next_token_onehot, img_embeds, next_token_hard, cond_logits, uncond_logits, cfg_logits, cond_hidden_states, uncond_hidden_states
