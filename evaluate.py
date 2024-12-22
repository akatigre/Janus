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

from janus.models import MultiModalityCausalLM, VLChatProcessor
from utils import set_seed, load_metadata, save_benchmark_images
from generate import generate_t2i
from tokenize_janus import tokenize_text
from decode import future_decode
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(theme=Theme({"logging.level.success": "green"})))]
)

log = logging.getLogger("rich")

@hydra.main(config_path="../configs", config_name="config")
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
    val_prompts, metadatas = load_metadata(cfg)
    categories = val_prompts.get("categories", None)
    N = len(val_prompts['prompts'])
    
    batch_size = 1
    per_prompt_images = []
    for start_idx in trange(0, N, batch_size):
        if cfg.benchmark.name=="geneval":
            prompts = val_prompts['prompts'][start_idx : start_idx + batch_size]
            names = val_prompts['name'][start_idx : start_idx + batch_size]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]
            metas = metadatas[start_idx: start_idx + batch_size]
            for save, metadata in zip(save_path[::4], metas[::4]):
                os.makedirs(save.parent, exist_ok=True)
                with open(os.path.join(save.parent, "metadata.jsonl"), "w") as fp:
                    json.dump(metadata, fp)

        elif cfg.benchmark.name=="dpgbench":
            prompts = val_prompts['prompts'][start_idx: start_idx + batch_size]
            names = val_prompts['name'][start_idx: start_idx + batch_size]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]

        elif cfg.benchmark.name=="mjhq":
            cats = categories[start_idx: start_idx + batch_size] if categories is not None else None
            gt_path = [Path(cfg.benchmark.outdirs).parent / 'root' / cat / name for cat, name in zip(cats, names)]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / cat / name for cat, name in zip(cats, names) if not (Path(cfg.benchmark.outdirs) / folder_name / cat / name).exists()]
            for save in save_path:
                os.makedirs(save.parent, exist_ok=True)
        else:
            raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        
        if not len(save_path):
            continue
        
        tokens, attention_mask = tokenize_text(vl_chat_processor, vl_gpt, prompts)
        tokens = tokens.to(device)
        attention_mask = attention_mask.to(device)
        input_embeds_prompt = vl_gpt.language_model.get_input_embeddings()(tokens) # bs, n_ctx, n_embed
        
        start_time = time.time()
        outputs = generate_t2i(
                        mmgpt = vl_gpt,
                        batch_size = len(prompts),
                        prompt_embeds = input_embeds_prompt,
                        attention_mask = attention_mask,
                        cfg = cfg,
                        **cfg
                    )
        
        if "yjk" in cfg.decode:
            perturbed_tokens, original_tokens = future_decode(
                outputs = outputs,
                cfg = cfg,
                mmgpt = vl_gpt,
                input_embeds_prompt = input_embeds_prompt,
            ) # batch_size, img_tokens
            
            
        else:
            all_tokens = outputs.generated_tokens # batch_size, img_tokens, 1
            
        dec = vl_gpt.gen_vision_model.decode_code(torch.cat([perturbed_tokens, original_tokens], dim=0).to(dtype=torch.int), shape=[batch_size, 8, 384 // 16, 384 // 16])
        dec = dec.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        images = np.zeros((2 * len(prompts), cfg.model_params.img_size, cfg.model_params.img_size, 3), dtype=np.uint8)
        images[:, :, :] = dec
    
        end_time = time.time()
        log.info(f"Time taken: {end_time - start_time} seconds")
        # save_benchmark_images(images, cfg, save_path)
        images = images.astype("uint8")
        from PIL import Image
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save("test_perturbed.png")
        pil_images[1].save("test_original.png")
        break
        # for save_at, image in zip(save_path, pil_images):
        #     save_at.parent.mkdir(parents=True, exist_ok=True)
        #     image.save(save_at)
        
    
if __name__=="__main__":
    main()