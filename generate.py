import torch
import numpy as np
from typing import Callable
from omegaconf import DictConfig
from janus.models import MultiModalityCausalLM


@torch.inference_mode()
def generate_t2i(
        mmgpt: MultiModalityCausalLM,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int = 16,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        temperature: float = 1.0,
        decode_func: Callable = None,
        cfg: DictConfig = None,
        **kwargs,
    ):

    # 1. Text to Image Input Processing
    generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        args = {
            "inputs_embeds": inputs_embeds, # 2, 13, 2048
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": outputs.past_key_values if i != 0 else None,
            "output_hidden_states": False,
        }

        outputs = mmgpt.language_model.model(
            **args
            )
        hidden_states = outputs.last_hidden_state # bsz, n_token in prompt, n_embed

        last_logits = mmgpt.gen_head(hidden_states[:, -1, :]) # 2 * batch_size, vocab_size (16384)

        uncond_logits, cond_logits = last_logits[0::2, :], last_logits[1::2, :]
        logits = decode_func(logit_cond = cond_logits, logit_uncond = uncond_logits, scale = cfg.cfg_scale, **kwargs)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[batch_size, 8, img_size//patch_size, img_size//patch_size])
    
    return dec