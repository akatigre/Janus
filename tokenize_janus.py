import PIL
import torch
from janus.models import VLChatProcessor, MultiModalityCausalLM
from janus.models.image_processing_vlm import VLMImageProcessor
from transformers import LlamaTokenizerFast
from typing import List

def tokenize_text(
    vl_chat_processor: VLChatProcessor, 
    vl_gpt: MultiModalityCausalLM,
    prompts: List,
    ):
    seq_ids = []
    batch_size = len(prompts)
    max_seq_len = 0
    for prompt in prompts:
        conversation = [
            {
                "role": "User",
                "content": prompt,
            },
            {
                "role": "Assistant", 
                "content": ""
            },
        ]
        
        sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations = conversation,
                    sft_format = vl_chat_processor.sft_format,
                    system_prompt="",
                )
        
        prompt = sft + vl_chat_processor.image_start_tag
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        seq_ids.append(input_ids)
        max_seq_len = max(max_seq_len, input_ids.shape[0])
    
    batched_input_ids = torch.full((batch_size, max_seq_len), vl_chat_processor.pad_id).long()
    batched_attention_mask = torch.zeros((batch_size, max_seq_len)).long()
    for i, input_ids in enumerate(seq_ids):
        seq_len = input_ids.shape[0]
        batched_input_ids[i, -seq_len:] = input_ids # left-padding
        batched_attention_mask[i, -seq_len:] = 1
    
    tokens = torch.zeros((batch_size * 2, max_seq_len), dtype=torch.int)
    attention_mask = batched_attention_mask.repeat_interleave(2, dim=0)
    tokens[0::2, 1:-1]  = vl_chat_processor.pad_id
    tokens[0::2, 0] = batched_input_ids[0, 0]
    tokens[0::2, -1] = batched_input_ids[0, -1]
    tokens[1::2] = batched_input_ids
    return tokens, attention_mask


def tokenize_image(
    vl_chat_processor: VLChatProcessor,
    vl_gpt: MultiModalityCausalLM,
    img_path: str,
    ):
    conversations = [
        {
            "role": "User",
            "content": "<image_placeholder>",
            "images": [img_path],
        },
        {
            "role": "Assistant", 
            "content": ""
        },
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversations,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=vl_chat_processor.system_prompt
    )

    # tokenize
    tokenizer: LlamaTokenizerFast = vl_chat_processor.tokenizer
    input_ids = tokenizer.encode(sft_format)
    input_ids = torch.LongTensor(input_ids)

    # add image tokens to the input_ids
    image_token_mask: torch.BoolTensor = input_ids == vl_chat_processor.image_id
    image_indices = image_token_mask.nonzero()
    input_ids, num_image_tokens = vl_chat_processor.add_image_token(
        image_indices=image_indices,
        input_ids=input_ids,
    )
    
    image_processor: VLMImageProcessor = vl_chat_processor.image_processor
    pil_img = PIL.Image.open(str(img_path))
    pil_img = pil_img.convert("RGB")
    pil_images = [pil_img]
    images_outputs = image_processor.preprocess(pil_images, return_tensors="pt")
    
    pixel_values = images_outputs.pixel_values.to(vl_gpt.dtype).to(vl_gpt.device)
    images_embeds = vl_gpt.aligner(vl_gpt.vision_model(pixel_values))

    return images_embeds