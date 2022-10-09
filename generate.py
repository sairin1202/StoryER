import torch.nn as nn
import torch
device = torch.device('cuda') 

def generate(model, inputs, tokenizer):
    if isinstance(model, nn.DataParallel):
        model = model.module  
    generated_ids = model.generate(
                    inputs["input_ids"].cuda(), 
                    attention_mask=inputs["attention_mask"].cuda(),
                    do_sample=True, 
                    decoder_start_token_id = 2, # eos token id
                    max_length=1000, 
                    top_p=0.9, 
                    top_k=0
                    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


