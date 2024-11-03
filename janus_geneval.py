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
import torch
from pathlib import Path
import PIL.Image
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


from utils import set_seed, gen_array
from llama_forward import change_llama_forward, change_llama_decoder_layer_forward, cfg_pag_forward
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


import logging
from rich.logging import RichHandler
from time import sleep
from rich.theme import Theme
from rich.console import Console

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(
        console=Console(theme=Theme({"logging.level.success": "green"}))
    )]
)

log = logging.getLogger("rich")

@hydra.main(config_path=".", config_name="geneval")
def main(cfg: DictConfig):
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(code_dir=".")
    )
    wandb.run.log_code("/home/server08/yoonjeon_workspace/MMAR/Janus/", include_fn=lambda path: path.endswith(".py"))
    model_path = cfg.model.path
    dtype = getattr(torch, cfg.model.dtype)
    device = cfg.model.device

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(dtype).to(device).eval()
    
    if cfg.layer_types=="all":
        layer_idxs = range(len(vl_gpt.language_model.model.layers))
    elif cfg.layer_types=="early":
        layer_idxs = range(len(vl_gpt.language_model.model.layers) // 3)
    elif cfg.layer_types=="middle":
        layer_idxs = range(len(vl_gpt.language_model.model.layers) // 3, 2 * len(vl_gpt.language_model.model.layers) // 3)
    elif cfg.layer_types=="late":
        layer_idxs = range(2 * len(vl_gpt.language_model.model.layers) // 3, len(vl_gpt.language_model.model.layers))

    enable_pag = cfg.pag_scale > 0.0
    enable_cfg = cfg.cfg_scale > 1.0
    enable_cd = cfg.cd_beta < 1.0
    log.info(f"Enable PAG: {enable_pag}, Enable CFG: {enable_cfg}, Enable CD: {enable_cd}")
    # Create folder to save images
    folder_name = "generated"
    if enable_pag: 
        folder_name += f"_pag:{cfg.pag_scale}_layer:{cfg.layer_types}"
        folder_name += "_add" if cfg.add else "_sub"
    if enable_cfg: folder_name += f"_cfg{cfg.cfg_scale}"
    if enable_cd: folder_name += f"_cd{cfg.cd_beta}"
    
    folder = os.path.join("outputs", folder_name)
    os.makedirs(folder, exist_ok=True)
    
    # Change attention layer structure to apply on PAG, CFG
    vl_gpt.language_model.model = change_llama_forward(vl_gpt.language_model.model)
    for idx, layer in enumerate(vl_gpt.language_model.model.layers):
        layer = change_llama_decoder_layer_forward(layer)
        layer.self_attn = cfg_pag_forward(layer.self_attn)
        if idx in layer_idxs:
            layer.self_attn.pag_layer = True
        else:
            layer.self_attn.pag_layer = False
     
    log.info("Layer structure changed successfully")
    # Set seed and load prompts
    set_seed(seed=cfg.seed)
    import json
    with open(cfg.metadata_file) as f:
        metadatas = [json.loads(line) for line in f]

    for p_idx, metadata in enumerate(metadatas):
        log.info(f"Prompt: {metadata['prompt']}")
        outpath = os.path.join(cfg.outdir, f"{p_idx:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        if len(os.listdir(sample_path)) == cfg.batch_size:
            continue
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        conversation = [
                {
                    "role": "User",
                    "content": metadata['prompt']
                },
                {
                    "role": "Assistant", 
                    "content": ""
                },
            ]

        visual_img, extras, generated_tokens = generate(
                                vl_gpt,
                                vl_chat_processor,
                                conversation = conversation,
                                **cfg
                                )
        
        # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # images *= 255.0
        # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # pil_images = [Image.fromarray(image) for image in images]
        
        sample_count = 0
        for sample in visual_img:
            sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1

        wandb_images = [wandb.Image(image, caption=prompt) for i, image in enumerate(pil_images)]

        wandb.log(
            {
                folder_name: wandb_images
            },
            step=p_idx
        )
        
        out_dir = Path(f"{sample_path}/geneval")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(extras, out_dir / "extras.pt")
        torch.save(generated_tokens, out_dir / "generated_tokens.pt")
        wandb.save(str(out_dir) + "/*", policy="end")


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    conversation: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_scale: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    cd_beta: float = 0.1,
    pag_scale: float = 5.0,
    add: bool = True,
    **kwargs,
):  
    enable_pag = pag_scale > 0.0
    enable_cfg = cfg_scale > 1.0
    enable_cd = cd_beta < 1.0
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
    prompt = sft_format + vl_chat_processor.image_start_tag
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    if enable_pag and enable_cfg:
        bs = parallel_size * 3
        tokens = torch.zeros((bs, len(input_ids)), dtype=torch.int).cuda()
        tokens[0::3, 1:-1]  = vl_chat_processor.pad_id
        tokens[0::3, 0] = input_ids[0]
        tokens[0::3, -1] = input_ids[-1]
        tokens[1::3] = input_ids[None].repeat(parallel_size, 1)
        tokens[2::3] = input_ids[None].repeat(parallel_size, 1)
    elif enable_cfg:
        bs = parallel_size * 2
        tokens = torch.zeros((bs, len(input_ids)), dtype=torch.int).cuda()
        tokens[0::2, 1:-1]  = vl_chat_processor.pad_id
        tokens[0::2, 0] = input_ids[0]
        tokens[0::2, -1] = input_ids[-1]
        tokens[1::2] = input_ids[None].repeat(parallel_size, 1)
    elif enable_pag:
        bs = parallel_size * 2
        tokens = torch.zeros((bs, len(input_ids)), dtype=torch.int).cuda()
        tokens[0::2] = input_ids[None].repeat(parallel_size, 1)
        tokens[1::2] = input_ids[None].repeat(parallel_size, 1)
    else:
        tokens = input_ids[None].repeat(parallel_size, 1).cuda()
    
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens) # bs, n_ctx, n_embed
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    extras = defaultdict(list)

    for i in range(image_token_num_per_image):
        args = {
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
            "past_key_values": outputs.past_key_values if i != 0 else None,
            "past_key_values_pag": outputs.past_key_values_ptb if i != 0 else None,
            "enable_pag": enable_pag,
            "enable_cfg": enable_cfg,
            "prefix_len": len(input_ids),
        }
            
        outputs = mmgpt.language_model.model(
            **args
            )
        hidden_states = outputs.last_hidden_state # 2 * parallel_size, n_token in prompt, n_embed
        logits = mmgpt.gen_head(hidden_states[:, -1, :]) # 2 * parallel_size, vocab_size (16384)

        if enable_pag and enable_cfg:
            logit_uncond = logits[0::3]
            logit_cond = logits[1::3]
            logit_pag = logits[2::3]
            if add:
                logits = (
                    logit_uncond
                    + cfg_scale * (logit_cond - logit_uncond) # making the null path with identity matrix
                    + pag_scale * (logit_cond - logit_pag) # identity matrix only on the last 4 layers -> adding softmax bias towards condition prefix
                )
            else:
                logits = (
                    logit_uncond 
                    + cfg_scale * (logit_cond - logit_uncond) 
                    - pag_scale * (logit_cond - logit_pag)
                )
        elif enable_pag:
            logit_cond = logits[0::2]
            logit_pag = logits[1::2]
            logits = logit_pag + pag_scale * (logit_cond - logit_pag) # cfg = (1 - gamma) / gamma
        
        elif enable_cfg:
            logit_uncond = logits[0::2]
            logit_cond = logits[1::2]
            logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
            
        if enable_cd:
            cd_beta = np.linspace(0.5, 0.9, image_token_num_per_image)[i]
            cutoff = torch.log(torch.tensor(cd_beta)) + logit_cond.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(logit_cond > cutoff, -float("inf"))

        topk_logits = logits.topk(100, dim=-1)
        extras["logit_hist_vals"].append(topk_logits.values)
        extras["logit_hist_inds"].append(topk_logits.indices)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # parallel_size, vocab_size -> get index of maximum token

        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        if enable_pag and enable_cfg:
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        elif enable_pag or enable_cfg:
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        else:
            next_token = next_token.view(-1)

        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    
    return visual_img, extras, generated_tokens
    
if __name__=="__main__":
    main()

    