import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList, UnbatchedClassifierFreeGuidanceLogitsProcessor, RepetitionPenaltyLogitsProcessor
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import logging

from functools import wraps
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from types import MethodType
import wandb
wandb.login(key="5295808ee2ec2b1fef623a0b1838c5a5c55ae8d1")

wandb.init(project = "CFG-Language")
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(theme=Theme({"logging.level.success": "green"})))]
)

logger = logging.getLogger("rich")
def soft_nll(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)

    
# specify the path to the model
model_path = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

cond_conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>\nDescribe the man wearing black suit in detail.",
        "images": ["images/image2.png"], #, "images/image2.png"],
    },
    {"role": "Assistant", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(cond_conversation)
prepare_inputs = vl_chat_processor(
    conversations=cond_conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)
negative_attn_mask = prepare_inputs.attention_mask.clone()
# # run image encoder to get the image embeddings 
input_ids_cond = prepare_inputs['input_ids']
input_ids_uncond = input_ids_cond.clone()
input_ids_uncond[:, 1:-1] = vl_chat_processor.pad_id
input_ids_uncond[input_ids_uncond < 0] = 0  # ignore the image embeddings
input_embeds_uncond = vl_gpt.language_model.get_input_embeddings()(input_ids_uncond)
bsz = input_ids_uncond.shape[0]

input_embeds_cond = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

gen_config = GenerationConfig(
    return_dict_in_generate=True,
    output_hidden_states=True,
    output_logits=True,
)
processors = LogitsProcessorList()
guidance_scale = 5
cfg_processor = UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    guidance_scale=guidance_scale,
                    model=vl_gpt.language_model,
                    unconditional_ids=input_ids_uncond,
                    use_cache=True
                )

# Define a wrapper for the method
def hooked_get_unconditional_logits(self, input_ids):
    if self.unconditional_context["first_pass"]:
        if self.unconditional_context["input_ids"] is None:
            self.unconditional_context["input_ids"] = input_ids[:, -1:]
        if self.unconditional_context["attention_mask"] is None:
            self.unconditional_context["attention_mask"] = torch.ones_like(
                self.unconditional_context["input_ids"], dtype=torch.long
            )
        input_ids = self.unconditional_context["input_ids"]
        attention_mask = self.unconditional_context["attention_mask"]
        self.unconditional_context["first_pass"] = False
    else:
        attention_mask = torch.cat(
            [
                self.unconditional_context["attention_mask"],
                torch.ones_like(input_ids[:, -1:], dtype=torch.long),
            ],
            dim=1,
        )
        if not self.unconditional_context["use_cache"]:
            input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
        else:
            input_ids = input_ids[:, -1:]
        self.unconditional_context["input_ids"] = input_ids
        self.unconditional_context["attention_mask"] = attention_mask
    out = self.model(
        input_ids,
        attention_mask=attention_mask,
        use_cache=self.unconditional_context["use_cache"],
        past_key_values=self.unconditional_context["past_key_values"],
        output_hidden_states=True
    )
    
    # Capture `hidden_states` from the intermediate result
    self.uncond_hidden = self.uncond_hidden if hasattr(self, "uncond_hidden") else ()
    self.uncond_hidden += (out.hidden_states,)
    self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

    return out.logits

# Replace the method with the hooked version
cfg_processor.get_unconditional_logits = MethodType(hooked_get_unconditional_logits, cfg_processor)
processors.append(cfg_processor)
processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))
#! Initial forward pass for initialization
outputs = vl_gpt.language_model.generate(
    inputs_embeds=input_embeds_cond,
    logits_processor=processors,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    generation_config=gen_config,
    max_new_tokens=512,
    do_sample=False, # do greedy decoding
    use_cache=True,
) 

answer = tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
logger.info(f"{prepare_inputs['sft_format'][0]} {answer}")

#! get uncond hidden states

uncond_hidden_prefix = cfg_processor.uncond_hidden[0][-1][:, :-1] # 512 (n_tokens) -> 25 (n_layers) -> bsz, n_token_per_iter, hidden_size
cond_hidden_prefix = outputs.hidden_states[0][-1][:, :-1] # 25 (n_layers) -> bsz, n_token_per_iter, hidden_size
prefix_len = cond_hidden_prefix.shape[1]
uncond_hidden = torch.stack([hidden_state[-1][:, -1] for hidden_state in cfg_processor.uncond_hidden], dim=1) # len(): 512 -> 25
cond_hidden = torch.stack([hidden_state[-1][:, -1] for hidden_state in outputs.hidden_states], dim=1) # len(): 25

n_tokens = cond_hidden.shape[1]
dim = vl_gpt.language_model.config.hidden_size # vl_gpt.language_model.config.vocab_size
vocab_size = vl_gpt.language_model.config.vocab_size

cond_epsilons = torch.nn.Parameter(torch.zeros((bsz, n_tokens, dim), device=vl_gpt.device, dtype=torch.bfloat16))
uncond_epsilons = torch.nn.Parameter(torch.zeros((bsz, n_tokens, dim), device=vl_gpt.device, dtype=torch.bfloat16))

topk = 10
stepsize = 5e-4
weight_decay = 1e-6
optimizer = torch.optim.AdamW([cond_epsilons, uncond_epsilons], lr=stepsize, weight_decay=weight_decay)

#! DEBUG 부터 망가져따
# input_embeds_cond -> input_ids -> lm_head -> logits
cond_logits = vl_gpt.language_model.lm_head(cond_hidden)
uncond_logits = vl_gpt.language_model.lm_head(uncond_hidden)
cond_logits = torch.nn.functional.log_softmax(cond_logits, dim=-1)
unconditional_logits = torch.nn.functional.log_softmax(uncond_logits, dim=-1)
scores_processed = guidance_scale * (cond_logits - unconditional_logits) + unconditional_logits
sequence = scores_processed.argmax(dim=-1)
answer = tokenizer.decode(sequence[0].cpu().tolist(), skip_special_tokens=True)
logger.info(f"{'-'*300} \n DEBUG: {answer}\n")

vocab_weights = vl_gpt.language_model.model.embed_tokens.weight
_, indices = torch.topk(scores_processed, k=topk, dim=-1)
mask_t = torch.zeros_like(scores_processed).scatter(2, indices, 1)
    
separate = False 
for i in range(100):
    cond_logits_perturbed = vl_gpt.language_model.lm_head(torch.cat([cond_hidden_prefix.detach(), cond_hidden.detach() + cond_epsilons], dim=1))
    uncond_logits_perturbed = vl_gpt.language_model.lm_head(torch.cat([uncond_hidden_prefix.detach(), uncond_hidden.detach() + uncond_epsilons], dim=1))
    cond_probs_perturbed = torch.nn.functional.log_softmax(cond_logits_perturbed, dim=-1)
    uncond_probs_perturbed = torch.nn.functional.log_softmax(uncond_logits_perturbed, dim=-1)
    if separate:
        cfg_perturbed = torch.cat([cond_probs_perturbed, uncond_probs_perturbed], dim=0)
    else: # mask from previous output / does not make sense
        cfg_perturbed = guidance_scale * (cond_probs_perturbed - uncond_probs_perturbed) + uncond_probs_perturbed
    
    if mask_t is not None:
        mask_t = mask_t.detach()
        sample_from = cfg_perturbed * mask_t + -1e10 * (1 - mask_t)
    
    labels = sample_from.argmax(dim=-1).detach()
    temperature = 1
    soft_input = (cfg_perturbed / 0.001) + cfg_perturbed - cfg_perturbed.detach()
    
    with torch.enable_grad():
        input_embeds = torch.matmul(soft_input, vocab_weights)
        outputs = vl_gpt.language_model(
            inputs_embeds=input_embeds, 
            attention_mask=prepare_inputs.attention_mask,
            return_dict=True, 
            output_hidden_states=True, 
            use_cache=False
            )
    if separate:
        cond_logits, uncond_logits = outputs.logits.chunk(2, dim=0)
        output_logits = guidance_scale * (cond_logits - uncond_logits) + uncond_logits
        input_cond_logits, input_uncond_logits = cfg_perturbed.chunk(2, dim=0)
        input_logits = guidance_scale * (input_cond_logits - input_uncond_logits) + input_uncond_logits
        
    else:
        output_logits = outputs.logits
        input_logits = cfg_perturbed
        shift_labels = labels[..., prefix_len : ].contiguous()
    shift_logits = output_logits[..., prefix_len - 1 : -1, : ].contiguous()
    input_logits = input_logits[:, prefix_len : ].contiguous()
    shift_labels = input_logits.argmax(dim=-1)
    
    loss_fnc = torch.nn.CrossEntropyLoss()
    ce_loss = loss_fnc(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    shift_logits = shift_logits * mask_t[:, prefix_len : ] + -1e10 * (1 - mask_t[:, prefix_len : ])
    soft_loss = soft_nll(shift_logits, input_logits)
    loss = ce_loss + soft_loss
    # loss = soft_loss
    if separate:
        loss = loss.mean()
    #! Soft NLL Loss
    loss.backward()
    wandb.log(
        {
            "loss": loss.item(),
            "answer": answer,
            "cond_gradient": torch.norm(
                cond_epsilons.grad
                ).detach().clone().data.float().cpu().numpy(),
            "uncond_gradient": torch.norm(
                uncond_epsilons.grad
                ).detach().clone().data.float().cpu().numpy()
        },
        step = i,
        )
    optimizer.step()
    optimizer.zero_grad()
    # cfg_perturbed = cfg_perturbed[:, prefix_len:] * mask + -1e10 * (1 - mask)
    sequence = cfg_perturbed[:, prefix_len:].argmax(dim=-1)
    answer = tokenizer.decode(sequence[0].cpu().tolist(), skip_special_tokens=True)
    noise = torch.normal(mean=0.01, std=0.01, size=uncond_hidden.size(), device='cuda', requires_grad=False, dtype=torch.bfloat16)
    cond_hidden = cond_hidden + noise * (1 - i / 100)
    uncond_hidden = uncond_hidden + noise * (1 - i / 100)
    logger.info(f"loss: {loss.item():.4f} | answer: {answer}")