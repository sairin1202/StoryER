from transformers import AdamW, get_linear_schedule_with_warmup 
from torch.optim import Adam

def get_optimizer(model, lr, train_steps, warmup_steps, weight_decay, adam_epsilon):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
                                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    return optimizer, lr_scheduler