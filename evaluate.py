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
    prompts = ["A yellow furry cat lying on a bed of red roses"]
    bsz = len(prompts)
    tokens, attention_mask = tokenize_text(vl_chat_processor, vl_gpt, prompts)
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    input_embeds_prompt = vl_gpt.language_model.get_input_embeddings()(tokens) # bs, n_ctx, n_embed
    original_tokens, cfg_logits, _ = generate_t2i(
                            mmgpt = vl_gpt,
                            batch_size = len(prompts),
                            prompt_embeds = input_embeds_prompt,
                            attention_mask = attention_mask,
                            do_langevin_dynamics = cfg.yjk.do_langevin_dynamics,
                            cfg = cfg,
                        ) 
    cfg_topk = torch.topk(cfg_logits, k=cfg.yjk.k_filter, dim=-1)[1] # Use cond_logits to filter out
    cfg_mask = torch.zeros_like(cfg_logits).scatter_(2, cfg_topk, 1)
    # pil_img = convert_to_pil(original_tokens, vl_gpt, bsz, cfg.model_params.img_size)
    # pil_img.save(f"debug/original_{cfg.cfg_scale}.png")
    
    if cfg.yjk.do_langevin_dynamics:
        name = f"debug/cfg{cfg.cfg_scale}_{cfg.yjk.optimizer}_lr{cfg.yjk.stepsize}_initnoise{cfg.yjk.start_noise}"
        name += "_hidden_bias" if cfg.yjk.use_hidden_state_bias else "_logit_bias"
        image_token_num_per_image = 576
        if cfg.yjk.use_hidden_state_bias:
            dim = 2048
           
        else:
            dim = 16384
        epsilons = torch.nn.Parameter(torch.zeros((bsz, image_token_num_per_image, dim)))  
        
        if cfg.yjk.optimizer == "Adam":
            optimizer = torch.optim.Adam([epsilons], lr=cfg.yjk.stepsize)
        elif cfg.yjk.optimizer == "AdamW":
            optimizer = torch.optim.AdamW([epsilons], lr=cfg.yjk.stepsize, weight_decay=cfg.yjk.weight_decay)
        elif cfg.yjk.optimizer == "SGD":
            optimizer = torch.optim.SGD([epsilons], lr=cfg.yjk.stepsize)
        else:
            raise ValueError(f"optimizer {cfg.yjk.optimizer} not supported.")
        
        best_loss = float('inf')
        temperature = 1.0

        for it in trange(cfg.yjk.update_iters):
            start_time = time.time()
            
            if cfg.yjk.use_hidden_state_bias:
                perturbed_logits = cfg_logits.to(device) + vl_gpt.gen_head(epsilons.to(device)) # 2 * batch_size, seq_len, vocab_size (16384)

            else:
                perturbed_logits = (cfg_logits + epsilons).to(device)
            
            selected_tokens = torch.argmax(perturbed_logits, dim=-1).squeeze(dim=-1)
            pil_img = convert_to_pil(selected_tokens, vl_gpt, bsz, cfg.model_params.img_size)
            pil_img.save(f"debug/p{prompts[0]}_iter{it}.png")
            
            if cfg_mask is None:
                soft_logits = (perturbed_logits / 0.001) - perturbed_logits.detach() + perturbed_logits    
            else:
                soft_logits = (perturbed_logits * cfg_mask.to(device) + -1e10 * (1 - cfg_mask.to(device))) / 0.001
            perturbed_probs = torch.softmax(soft_logits / temperature, dim=-1)
            weights = vl_gpt.gen_embed.weight # mmgpt tensors are inference mode, so clone into regular tensor that records gradient
            img_embeds = torch.matmul(torch.cat([perturbed_probs, perturbed_probs], dim=0), weights) # 2 * batch_size, seq_len, 2048
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
            last_logits = vl_gpt.gen_head(hidden_states) # 2 * batch_size, seq_len, vocab_size (16384)
            uncond_logits, cond_logits = last_logits[0::2, :], last_logits[1::2, :]
            prefix_len = input_embeds_prompt.shape[1]
            cfg_logits_new = cfg_decode(logit_cond = cond_logits, logit_uncond = uncond_logits, scale = cfg.cfg_scale)[:, prefix_len:]
            
            cfg_logits_new = cfg_logits_new * cfg_mask.to(device) + -1e10 * (1 - cfg_mask.to(device))
            # loss_fn = torch.nn.CrossEntropyLoss()
            # Note that < n predicts n, therefore input / labels should be shifted
            # loss = loss_fn(cfg_logits_new[:, : -1].view(-1, 16384), original_tokens[:, 1 : ].detach().view(-1).long())
            loss = soft_nll(cfg_logits_new[:, : -1], cfg_logits[:, 1:].to(device))
            loss.backward()
            
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                                     
            noise = torch.normal(mean=0.01, std=0.01, size=epsilons.shape, requires_grad=False) # reduce per iter
            epsilons.data = epsilons.data + noise
            log.info(f"loss: {loss.item()} Best loss: {best_loss}")
            end_time = time.time()
            log.info(f"Time taken: {end_time - start_time} seconds")

                
          
if __name__=="__main__":
    main()