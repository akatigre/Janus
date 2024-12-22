import torch
import numpy as np
from typing import Callable
from omegaconf import DictConfig
from janus.models import MultiModalityCausalLM
from decode import cfg_decode, rollout

class Outputs:
    def __init__(self, generated_tokens, output_hidden_states, output_logits, output_selected_probs, past_key_values):
        self.generated_tokens = generated_tokens
        self.output_hidden_states = output_hidden_states
        self.output_logits = output_logits
        self.output_selected_probs = output_selected_probs
        self.past_key_values = past_key_values

@torch.inference_mode()
def generate_t2i(
        mmgpt: MultiModalityCausalLM,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int = 16,
        image_token_num_per_image: int = 576,
        temperature: float = 1.0,
        cfg: DictConfig = None,
        rollout_len: int = 5,
        **kwargs,
    ):

    generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()
    output_selected_probs = torch.zeros((image_token_num_per_image, batch_size, 1)).cuda()
    output_hidden_states = torch.zeros((image_token_num_per_image, 2 * batch_size, 2048)).cuda()
    output_logits = torch.zeros((image_token_num_per_image, 2 * batch_size, 16384)).cuda()
    
    
    inputs_embeds = prompt_embeds
    for i in range(image_token_num_per_image):    
        if cfg.decode == "yjk_lookahead":
            next_token = rollout(
                mmgpt = mmgpt,
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                batch_size = batch_size,
                rollout_len = rollout_len,
                cfg = cfg,
                temperature = temperature,
            )
            
        else:
            args = {
                "inputs_embeds": inputs_embeds, # 8, 13, 2048
                "attention_mask": attention_mask, # 8, 13 = padding mask to indicate padding tokens
                "use_cache": True,
                "past_key_values": outputs.past_key_values if i != 0 else None,
                "output_hidden_states": False,
            }
            outputs = mmgpt.language_model.model(
                **args
                )
            hidden_states = outputs.last_hidden_state # 2 * bsz, inputs_embeds.shape[1], n_embed
            last_logits = mmgpt.gen_head(hidden_states[:, -1, :]) # 2 * batch_size, vocab_size (16384)
            output_hidden_states[i] = hidden_states[:, -1, :] # last hidden state of each token generation step
            
            uncond_logits, cond_logits = last_logits[0::2, :], last_logits[1::2, :]
            output_logits[i, 0::2, :] = uncond_logits
            output_logits[i, 1::2, :] = cond_logits
            logits = cfg_decode(logit_cond = cond_logits, logit_uncond = uncond_logits, scale = cfg.cfg_scale, **kwargs)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # batch_size, 1
            
        selected_indices = next_token.squeeze(-1)  # Get the index of the selected token
        selected_probs = probs.gather(-1, selected_indices.unsqueeze(-1)) # batch_size, 1
        output_selected_probs[i] = selected_probs
        
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)  # 2 * batch_size
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token) # 2 * batch_size, 2048
        inputs_embeds = img_embeds.unsqueeze(dim=1) # 2 * batch_size, 1, 2048
        
    return Outputs(
        generated_tokens = generated_tokens, 
        output_hidden_states = output_hidden_states, 
        output_logits = output_logits,
        output_selected_probs = output_selected_probs,
        past_key_values = outputs.past_key_values,
        )