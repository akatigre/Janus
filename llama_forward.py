import types
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from transformers.utils import ModelOutput
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaFlashAttention2, apply_rotary_pos_emb, LlamaSdpaAttention, repeat_kv
from rich.logging import RichHandler
import logging
# Configure the logger to use RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)

@dataclass
class Output(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values_ptb: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def change_llama_forward(model: LlamaModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        past_key_values_pag: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_pag = DynamicCache.from_legacy_cache(past_key_values_pag)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        next_decoder_cache_pag = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    past_key_values_pag,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values, # Dynamic
                    past_key_value_pag=past_key_values_pag, # DynamicCache lives until here
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache and not kwargs["enable_pag"]:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            elif use_cache and kwargs["enable_pag"]:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache_pag = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        next_cache_pag = next_decoder_cache_pag if use_cache else None

        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
            if next_cache_pag is not None:
                next_cache_pag = next_cache_pag.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return Output(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            past_key_values_ptb=next_cache_pag,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    model.forward = types.MethodType(forward, model)
    return model

def change_llama_decoder_layer_forward(decoder_layer: LlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_value_pag: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_value_pag=past_key_value_pag,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if kwargs["enable_pag"]:
            hidden_states, self_attn_weights, present_key_value, present_key_value_pag = outputs
        else:
            hidden_states, self_attn_weights, present_key_value = outputs
            present_key_value_pag = None
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value, present_key_value_pag)

        return outputs
    decoder_layer.forward = types.MethodType(forward, decoder_layer)
    return decoder_layer
  
@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask

def attn_with_flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask_mod: callable):
    L = q.size(2) # sequence length
    S = k.size(2)

    block_mask = create_block_mask_cached(mask_mod, 1, 1, L, S, device=q.device)
    output = flex_attention(
        q,
        k,
        v,
        block_mask = block_mask
    )
    return output
 
def cfg_pag_forward(attention: LlamaSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_value_pag: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        enable_pag: Optional[bool] = False,
        enable_cfg: Optional[bool] = False,
        prefix_len: Optional[int] = 50,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False
        if not enable_cfg and enable_pag:
            hidden_states_cond, hidden_states_ptb = hidden_states[0::2], hidden_states[1::2]
        elif enable_cfg and enable_pag:
            hidden_states_uncond, hidden_states_cond, hidden_states_ptb = hidden_states[0::3], hidden_states[1::3], hidden_states[2::3]
            if self.pag_layer:
                hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond], dim=0)
            else:
                hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond, hidden_states_ptb], dim=0)
        elif enable_cfg and not enable_pag:
            hidden_states_uncond, hidden_states_cond = hidden_states[0::2], hidden_states[1::2] # uncond, cond => cfg or cond
            hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond], dim=0)
        else:
            pass
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.

        is_causal = True if q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        if not output_attentions:
            attn_weights = None

        
        if enable_pag:
            bsz_ptb, q_len, _ = hidden_states_ptb.shape
            query_states_ptb = self.q_proj(hidden_states_ptb)
            key_states_ptb = self.k_proj(hidden_states_ptb)
            value_states_ptb = self.v_proj(hidden_states_ptb)

            query_states_ptb = query_states_ptb.view(bsz_ptb, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states_ptb = key_states_ptb.view(bsz_ptb, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states_ptb = value_states_ptb.view(bsz_ptb, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states_ptb, key_states_ptb = apply_rotary_pos_emb(query_states_ptb, key_states_ptb, cos, sin)

            if past_key_value_pag is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_ptb, value_states_ptb = past_key_value_pag.update(key_states_ptb, value_states_ptb, self.layer_idx, cache_kwargs)
        
        ########### PAG path ############
        if enable_pag and self.pag_layer:    
                
            key_states_ptb = repeat_kv(key_states_ptb, self.num_key_value_groups)
            value_states_ptb = repeat_kv(value_states_ptb, self.num_key_value_groups)

            k_len = key_states_ptb.size(2)
            attention_mask = torch.zeros((q_len, k_len), device=query_states_ptb.device, dtype=query_states_ptb.dtype)
            attention_mask[ : , prefix_len : -1] = float("-inf")
            # expand the mask to match the attention weights shape
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # Add batch and num_heads dimensions
            attn_output_ptb = torch.nn.functional.scaled_dot_product_attention(
                query_states_ptb,
                key_states_ptb,
                value_states_ptb,
                attention_mask,
                dropout_p = self.attention_dropout if self.training else 0.0,
                is_causal = True if q_len > 1 else False
            )
            attn_output_ptb = attn_output_ptb.transpose(1, 2).contiguous()
            attn_output_ptb = attn_output_ptb.view(bsz_ptb, q_len, -1)

            attn_output = torch.cat([attn_output, attn_output_ptb], dim=0)
            attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)

        if enable_cfg and enable_pag:
            attn_uncond, attn_cond, attn_ptb = attn_output.chunk(3)
            attn_output_ = torch.zeros_like(attn_output)
            for i in range(attn_output.shape[0] // 3):
                attn_output_[3 * i] = attn_uncond[i]
                attn_output_[3 * i + 1] = attn_cond[i]
                attn_output_[3 * i + 2] = attn_ptb[i]

            
        elif enable_cfg or enable_pag:
            attn_uncond, attn_cond = attn_output.chunk(2)
            attn_output_ = torch.zeros_like(attn_output)
            for i in range(attn_output.shape[0] // 2):
                attn_output_[2 * i] = attn_uncond[i]
                attn_output_[2 * i + 1] = attn_cond[i]

        if enable_pag:
            outputs = [attn_output_, attn_weights, past_key_value, past_key_value_pag]
        else:
            outputs = [attn_output_, attn_weights, past_key_value]
        return outputs
    
    attention.forward = types.MethodType(forward, attention)
    return attention