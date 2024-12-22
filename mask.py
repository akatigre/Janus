import torch

def create_attn_mask(
    total_input_ids,
    rm_pad_in_image = True,
    pad_id = 0,
    soi_id = 1,
    eoi_id = 2,
    ):
    N, L = total_input_ids.shape
    
    # Masks to identify different types of tokens
    is_padding = total_input_ids == pad_id
    is_start_image = total_input_ids == soi_id
    is_end_image = total_input_ids == eoi_id

    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image # image mask
    is_text = ~(in_image_segment) # prefix (text) mask
    
    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(total_input_ids.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]

    is_text_image = is_text | in_image_segment

    mask_text_image_bi = is_text_image[:, :, None] * is_text_image[:, None, :]
    if rm_pad_in_image:
        sid_img = torch.where(total_input_ids == soi_id)[1]
        for i in range(mask_text_image_bi.shape[0]):
            pad_end_idx = torch.where(total_input_ids[i] == pad_id)
            if len(pad_end_idx[0]) != 0:
                pad_end_idx = pad_end_idx[0][-1]
                mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
            id_padding = torch.where(is_padding[i] == True)
            mask_text_image_bi[i][sid_img[i]:, id_padding[0]] = 0
            
    mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]
    attention_mask = mask_text.unsqueeze(1)
    return attention_mask