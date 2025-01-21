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
        cfg_scale: float = 5.0,
        **kwargs,
    ): 
    
    generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()
    
    output_cond_logits = torch.zeros((batch_size, image_token_num_per_image, 16384))
    output_uncond_logits = torch.zeros((batch_size, image_token_num_per_image, 16384))
    
    output_cond_hidden = torch.zeros((batch_size, image_token_num_per_image, 2048))
    output_uncond_hidden = torch.zeros((batch_size, image_token_num_per_image, 2048))
    
    inputs_embeds = prompt_embeds
    outputs = None
    for i in trange(image_token_num_per_image):    
        with torch.inference_mode():
            inputs_embeds, outputs, hidden_states, last_logits, next_token_hard = soft_forward(
                mmgpt = mmgpt,
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                outputs = outputs,
                soft = False,
                temperature = temperature,
                cfg_scale = cfg_scale,
                i = i
            )
        
        uncond_logits, cond_logits = last_logits[0::2, :].detach().clone().cpu(), last_logits[1::2, :].detach().clone().cpu()
        uncond_hidden_states, cond_hidden_states = hidden_states[0::2].detach().clone().cpu(), hidden_states[1::2].detach().clone().cpu()
        generated_tokens[:, i:i+1] = next_token_hard.squeeze(dim=-1).detach().clone()
        output_cond_logits[:, i:i+1] = cond_logits.detach().clone().cpu()
        output_uncond_logits[:, i:i+1] = uncond_logits.detach().clone().cpu()
        output_cond_hidden[:, i:i+1] = cond_hidden_states.detach().clone().cpu()
        output_uncond_hidden[:, i:i+1] = uncond_hidden_states.detach().clone().cpu()
    return generated_tokens, output_cond_logits, output_uncond_logits, output_cond_hidden, output_uncond_hidden
    
def soft_forward(
            mmgpt: MultiModalityCausalLM, 
            inputs_embeds,
            attention_mask,
            outputs,
            soft: bool = True, 
            temperature: float = 1.0, 
            cfg_scale: float = 5.0,
            i: int = 0
            ):
    
    args = {
        "inputs_embeds": inputs_embeds, # 8, 13, 2048
        "attention_mask": attention_mask, # 8, 13 = padding mask to indicate padding tokens
        "use_cache": True,
        "past_key_values": outputs.past_key_values if i!=0 else None, 
        "output_hidden_states": False,
    }
    
    outputs = mmgpt.language_model.model(
                **args
            )
        
    if not i:
        hidden_states = outputs.last_hidden_state[:, -1, :].unsqueeze(dim=1) # for prefix, remove
    else:
        hidden_states = outputs.last_hidden_state

    assert hidden_states.dim() == 3, "hidden_states should be of shape (2 * batch_size, seq_len, 2048)"
    bsz = hidden_states.shape[0] // 2
    last_logits = mmgpt.gen_head(hidden_states) # 2 * batch_size, seq_len, vocab_size (16384)
    uncond_logits, cond_logits = last_logits[0::2, :], last_logits[1::2, :]
    cfg_logits = cfg_decode(logit_cond = cond_logits, logit_uncond = uncond_logits, scale = cfg_scale)
    cfg_probs = torch.softmax(cfg_logits / temperature, dim=-1)
    # next_token_hard = torch.multinomial(cfg_probs.view(-1, 16384), num_samples=1).view(bsz, -1) # batch_size, seq_len
    next_token_hard = cfg_probs.argmax(dim=-1).view(bsz, -1)
    next_token_onehot = torch.nn.functional.one_hot(next_token_hard, num_classes=16384) # batch_size, seq_len, 16384
    next_token_onehot = next_token_onehot - cfg_probs.detach() + cfg_probs
    weights = mmgpt.gen_embed.weight # mmgpt tensors are inference mode, so clone into regular tensor that records gradient
    img_embeds = torch.matmul(torch.cat([next_token_onehot, next_token_onehot], dim=0), weights) # 2 * batch_size, seq_len, 2048
    img_embeds = mmgpt.gen_aligner(img_embeds) # 2 * batch_size, seq_len, 2048
    return img_embeds, outputs, hidden_states, last_logits, next_token_hard