import os
import torch
from transformers import T5EncoderModel, AutoModelForCausalLM, AutoTokenizer

# load text-encoder
def load_text_encoder(text_encoder_dir, device, weight_dtype):
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
    if 'gemma' in text_encoder_dir:
        tokenizer.padding_side = "right"
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_dir, attn_implementation="flash_attention_2", device_map='cpu', torch_dtype=weight_dtype
        ).get_decoder()
    elif 't5' in text_encoder_dir:
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_dir, attn_implementation="sdpa", device_map='cpu', torch_dtype=weight_dtype
        )
    else: 
        raise NotImplementedError
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.eval().to(device=device, dtype=weight_dtype)
    
    return text_encoder, tokenizer
    
def encode_prompt(tokenizer, text_encoder, device, weight_dtype, captions, use_last_hidden_state, max_seq_length=256):
    text_inputs = tokenizer(
        captions,
        padding='max_length',
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        results = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )

        if use_last_hidden_state:
            prompt_embeds = results.last_hidden_state
        else:   # from Imagen paper
            prompt_embeds = results.hidden_states[-2]

    return prompt_embeds, prompt_masks

