# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import time
from PIL import Image
import json
from pathlib import Path

import torch
import hydra
import logging
import numpy as np
from tqdm import trange
from rich.theme import Theme
from rich.console import Console
from rich.logging import RichHandler
from omegaconf import DictConfig

from transformers import AutoModelForCausalLM
from torchvision.utils import make_grid, save_image
from janus.models import MultiModalityCausalLM, VLChatProcessor
from utils import set_seed, load_metadata, convert_to_pil
from generate import generate_t2i
from tokenize_janus import tokenize_text
from decode import cfg_decode
from torch.nn import functional as F
            
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(theme=Theme({"logging.level.success": "green"})))]
)

log = logging.getLogger("rich")

def soft_nll(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)

@hydra.main(config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    log.info(f"Set seed {cfg.seed}")
    model_params = cfg.model_params
    assert model_params.model_name == "Janus", "Model name should be Janus"
    
    folder_name = f"generated/{cfg.model_params.model_name}/{cfg.decode}{cfg.cfg_scale}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = cfg.model_params.model.path
    dtype = getattr(torch, cfg.model_params.model.dtype)
    
    torch.set_default_dtype(dtype)
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(dtype).to(device).eval()
    torch.set_default_dtype(dtype)
    
    # Prepare prompts and metadata
    # val_prompts, metadatas = load_metadata(cfg)
    # categories = val_prompts.get("categories", None)
    # N = len(val_prompts['prompts'])
    
    # batch_size = 4
    # per_prompt_images = []
    # for start_idx in trange(0, N, batch_size):
        # start_idx = cfg.prompt_idx 
        # if cfg.benchmark.name=="geneval":
        #     prompts = val_prompts['prompts'][start_idx : start_idx + batch_size]
        #     names = val_prompts['name'][start_idx : start_idx + batch_size]
        #     save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]
        #     metas = metadatas[start_idx: start_idx + batch_size]
        #     for save, metadata in zip(save_path[::4], metas[::4]):
        #         os.makedirs(save.parent, exist_ok=True)
        #         with open(os.path.join(save.parent, "metadata.jsonl"), "w") as fp:
        #             json.dump(metadata, fp)

        # elif cfg.benchmark.name=="dpgbench":
        #     prompts = val_prompts['prompts'][start_idx: start_idx + batch_size]
        #     names = val_prompts['name'][start_idx: start_idx + batch_size]
        #     save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]

        # elif cfg.benchmark.name=="mjhq":
        #     cats = categories[start_idx: start_idx + batch_size] if categories is not None else None
        #     gt_path = [Path(cfg.benchmark.outdirs).parent / 'root' / cat / name for cat, name in zip(cats, names)]
        #     save_path = [Path(cfg.benchmark.outdirs) / folder_name / cat / name for cat, name in zip(cats, names) if not (Path(cfg.benchmark.outdirs) / folder_name / cat / name).exists()]
        #     for save in save_path:
        #         os.makedirs(save.parent, exist_ok=True)
        # else:
        #     raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        
        # if not len(save_path):
        #     continue
    # prompts = ["A yellow furry cat lying on a bed of red roses"]
    prompts = ["A rabbit and a cat playing chess"]
    bsz = len(prompts)
    tokens, attention_mask = tokenize_text(vl_chat_processor, vl_gpt, prompts)
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    input_embeds_prompt = vl_gpt.language_model.get_input_embeddings()(tokens) # bs, n_ctx, n_embed
    prefix_len = input_embeds_prompt.shape[1]
    generated_tokens, output_cond_logits, output_uncond_logits, output_cond_hidden, output_uncond_hidden = generate_t2i(
                            mmgpt = vl_gpt,
                            batch_size = len(prompts),
                            prompt_embeds = input_embeds_prompt,
                            attention_mask = attention_mask,
                            cfg_scale = cfg.cfg_scale,
    )
    cfg_original = cfg_decode(logit_cond = output_cond_logits, logit_uncond = output_uncond_logits, scale = cfg.cfg_scale)
    # topk = torch.topk(cfg_original, k=cfg.yjk.k_filter, dim=-1)[1]
    # original_mask = (torch.zeros_like(cfg_original) - 1e10).scatter_(2, topk, 1)
    os.makedirs(f"debug/{prompts[0]}", exist_ok=True)
    pil_img = convert_to_pil(generated_tokens, vl_gpt, bsz, cfg.model_params.img_size)
    pil_img.save(f"debug/{prompts[0]}/original_{cfg.cfg_scale}.png")
    
    if cfg.yjk.do_langevin_dynamics:
        name = f"debug/{prompts[0]}/cfg{cfg.cfg_scale}_{cfg.yjk.optimizer}_lr{cfg.yjk.stepsize}_initnoise{cfg.yjk.start_noise}_topk{cfg.yjk.k_filter}"
        name += "_hidden_bias" if cfg.yjk.use_hidden_state_bias else "_logit_bias"
        image_token_num_per_image = 576
        if cfg.yjk.use_hidden_state_bias:
            dim = 2048
        else:
            dim = 16384
            
        cond_epsilons = torch.nn.Parameter(torch.zeros((bsz, image_token_num_per_image, dim)))  
        uncond_epsilons = torch.nn.Parameter(torch.zeros((bsz, image_token_num_per_image, dim)))
        if cfg.yjk.optimizer == "Adam":
            optimizer = torch.optim.Adam([cond_epsilons, uncond_epsilons], lr=cfg.yjk.stepsize)
        elif cfg.yjk.optimizer == "AdamW":
            optimizer = torch.optim.AdamW([cond_epsilons, uncond_epsilons], lr=cfg.yjk.stepsize, weight_decay=cfg.yjk.weight_decay)
        elif cfg.yjk.optimizer == "SGD":
            optimizer = torch.optim.SGD([cond_epsilons, uncond_epsilons], lr=cfg.yjk.stepsize)
        else:
            raise ValueError(f"optimizer {cfg.yjk.optimizer} not supported.")
        
        best_loss = float('inf')
        temperature = 1.0
        
        loss_fn = torch.nn.CrossEntropyLoss()
        for it in trange(cfg.yjk.update_iters):
            start_time = time.time()
            
            if cfg.yjk.use_hidden_state_bias:
                perturbed_logits = torch.cat(
                    [
                        vl_gpt.gen_head((output_cond_hidden + cond_epsilons).to(device)),
                        vl_gpt.gen_head((output_uncond_hidden + uncond_epsilons).to(device))
                    ], dim=0)
                # perturbed_logits = cfg_logits.to(device) + vl_gpt.gen_head(epsilons.to(device)) # 2 * batch_size, seq_len, vocab_size (16384)
            else:
                perturbed_logits = torch.cat(
                    [
                        (output_cond_logits + cond_epsilons).to(device),
                        (output_uncond_logits + uncond_epsilons).to(device)
                    ], dim=0)
            
            soft_logits = (perturbed_logits / 0.001) - perturbed_logits.detach() + perturbed_logits    
            perturbed_probs = torch.softmax(soft_logits / temperature, dim=-1) # 2 * batch_size, seq_len, vocab_size (16384)
            uncond, cond = perturbed_probs[0::2, :, :], perturbed_probs[1::2, :, :]
            cfg_logits_input = cfg_decode(logit_cond = cond, logit_uncond = uncond, scale = cfg.cfg_scale)
            cfg_probs = torch.softmax(cfg_logits_input / temperature, dim=-1)
            output_ids = cfg_probs.argmax(dim=-1).view(bsz, -1) # * model input maximum tokens
            
            weights = vl_gpt.gen_embed.weight # mmgpt tensors are inference mode, so clone into regular tensor that records gradient
            # img_embeds = torch.matmul(torch.cat([perturbed_probs, perturbed_probs], dim=0), weights) # 2 * batch_size, seq_len, 2048
            img_embeds = torch.matmul(perturbed_probs, weights)
            img_embeds = vl_gpt.gen_aligner(img_embeds) # 2 * batch_size, seq_len, 2048

            args = {
                "inputs_embeds": torch.cat([input_embeds_prompt.detach(), img_embeds], dim=1), # 8, 13, 2048
                "attention_mask": attention_mask, # 8, 13 = padding mask to indicate padding tokens
                "use_cache": False,
                "past_key_values": None,
                "output_hidden_states": False,
            }

            outputs = vl_gpt.language_model.model(
                **args
            )
            hidden_states = outputs.last_hidden_state
            last_logits = vl_gpt.gen_head(hidden_states)[:, prefix_len:] # 2 * batch_size, seq_len, vocab_size (16384)
            
            uncond_logits, cond_logits = last_logits[0::2, :], last_logits[1::2, :]
            # output_ids_cond = cond.argmax(dim=-1).view(bsz, -1)
            # cond_loss = loss_fn(cond_logits[:, : -1].view(-1, 16384), output_ids_cond[:, 1 : ].detach().view(-1).long())
            # output_ids_uncond = uncond.argmax(dim=-1).view(bsz, -1)
            # uncond_loss = loss_fn(uncond_logits[:, : -1].view(-1, 16384), output_ids_uncond[:, 1 : ].detach().view(-1).long())
            cfg_logits_output = cfg_decode(logit_cond = cond_logits, logit_uncond = uncond_logits, scale = cfg.cfg_scale) # * model output 
            # mask = torch.zeros_like(cfg_original).scatter_(2, topk, 1)[:, 1 : ].to(device)
            loss = loss_fn((cfg_logits_output[:, : -1]).view(-1, 16384), output_ids[:, 1 : ].detach().view(-1).long())
            # loss = cond_loss + uncond_loss

            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_perturbed = perturbed_logits.detach()
                        
            noise = torch.normal(mean=0.01, std=0.01, size=cond_epsilons.shape, requires_grad=False) # reduce per iter
            uc_noise = torch.normal(mean=0.01, std=0.01, size=uncond_epsilons.shape, requires_grad=False)
            cond_epsilons.data = cond_epsilons.data + noise
            uncond_epsilons.data = uncond_epsilons.data + uc_noise
            end_time = time.time()
            log.info(f"Loss: {loss.item()} Best loss: {best_loss} Time taken: {end_time - start_time}") # Cond loss: {cond_loss.item()} Uncond loss: {uncond_loss.item()} 
            
        uncond, cond = best_perturbed[0::2, :, :], best_perturbed[1::2, :, :]
        cfg_logits = cfg_decode(logit_cond = cond, logit_uncond = uncond, scale = cfg.cfg_scale)
        cfg_probs = torch.softmax(cfg_logits / temperature, dim=-1)
        selected_tokens_sample = torch.multinomial(cfg_probs.view(-1, 16384), num_samples=1).view(bsz, -1) # batch_size, seq_len
        selected_tokens_greedy = cfg_probs.argmax(dim=-1).view(bsz, -1)
        pil_img = convert_to_pil(selected_tokens_sample, vl_gpt, bsz, cfg.model_params.img_size)
        pil_img.save(f"{name}_sampling.png")
        pil_img = convert_to_pil(selected_tokens_greedy, vl_gpt, bsz, cfg.model_params.img_size)
        pil_img.save(f"{name}_greedy.png")
          

if __name__=="__main__":
    main()