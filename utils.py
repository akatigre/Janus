
import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

from torchvision.utils import make_grid
from PIL import Image, ImageFont, ImageDraw
from transformers import set_seed as hf_set_seed


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)


def load_metadata(cfg):
    """
        load text_prompts and metadatas repeated by the required number of generations for each benchmark dataset
    """
    val_prompts = defaultdict(list)
    
    prompt_path = cfg.benchmark.prompts
    if cfg.benchmark.name=="dpgbench":
        prompt_lists = sorted(os.listdir(prompt_path))
        for p in prompt_lists:
            full_path = os.path.join(prompt_path, p)
            with open(full_path, 'r') as f:
                line = f.read().splitlines()[0]
            val_prompts["name"].extend([p.replace("txt", "png")] * cfg.benchmark.batch)
            val_prompts["prompts"].extend([line] * cfg.benchmark.batch)
        metadatas = None
        
    elif cfg.benchmark.name=="geneval":
        with open(prompt_path) as f:
            metadatas = [json.loads(line) for line in f for _ in range(cfg.benchmark.batch)]
        val_prompts["prompts"] = [metadata['prompt'] for metadata in metadatas]
        val_prompts["name"] = [f"{idx:0>5}/{img_idx:05}.png" for idx in range(len(val_prompts["prompts"])) for img_idx in range(cfg.benchmark.batch)]
        
    elif cfg.benchmark.name=="mjhq":
        with open(prompt_path, "r") as f:
            metadatas = json.load(f)
        file_names = sorted(list(metadatas.keys()))
        
        val_prompts["name"] = [file_name + ".jpg" for file_name in file_names]
        val_prompts["prompts"] = [metadatas[filename]["prompt"] for filename in file_names]
        val_prompts["categories"] = [metadatas[filename]["category"] for filename in file_names]
        
    else:
        raise NotImplementedError(f"Unknown benchmark name: {cfg.benchmark.name}")
    return val_prompts, metadatas

def save_image(images, gt_images, prompt, cfg, teacher_force_upto):
    w, h = images.size
    
    width = 2 * w
    font_path = "/root/.local/share/fonts/D2CodingLigatureNerdFontMono-Regular.ttf"  # Update this with your font file path
    font_size = 30  # Adjust font size as needed
    font = ImageFont.truetype(font_path, font_size)
    text_height = 100  # Adding some padding
    height = h + text_height
    comb_image = Image.new('RGB', (width, height), (255, 255, 255))
    comb_image.paste(gt_images, (0, 0))
    comb_image.paste(images, (w, 0))
    
    draw = ImageDraw.Draw(comb_image)
    img_type = f"Decoding: {cfg.decode}" if not cfg.teacher_force else f"Reconstruction Upto {teacher_force_upto * 100:.0f}%"
    draw.text((w + 20, h + 20), img_type, fill=(0, 0, 0), font=font)
    w = draw.textlength(prompt, font=font)
    x_position = (width - w) // 2
    draw.text((x_position, h + 55), prompt, fill=(0, 0, 0), font=font)
    return comb_image

def save_benchmark_images(images, cfg, save_path):
    if cfg.benchmark.name=="dpgbench":
        per_prompt_images.extend([image for image in images])
        for img_idx in range(0, len(per_prompt_images), cfg.benchmark.batch):
            images = make_grid(per_prompt_images[img_idx: img_idx + cfg.benchmark.batch], nrow=2)
            images = images.astype('uint8')
            images = Image.fromarray(images)
            save_path[img_idx].parent.mkdir(parents=True, exist_ok=True)
            images.save(save_path[img_idx])
        per_prompt_images = []
    else:
        images = images.astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for save_at, image in zip(save_path, pil_images):
            save_at.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_at)
            
            
def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1), 1)
    onehot.to(tensor.device)
    return onehot

def convert_to_pil(tokens, vl_gpt, bsz, img_size):
    shape = [bsz, 8, 384 // 16, 384 // 16]
    dec = vl_gpt.gen_vision_model.decode_code(tokens.to(dtype=torch.int), shape=shape)
    dec = dec.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    images = np.zeros((bsz, img_size, img_size, 3), dtype=np.uint8)
    # Populate the images array with dec values
    images[:, :, :] = dec
    if images.dtype != np.uint8:
        images = (images * 255).clip(0, 255).astype("uint8")
    # Convert to PyTorch tensor and normalize
    pil_img = Image.fromarray(images[0]) # pil image: b, h w, c
    return pil_img