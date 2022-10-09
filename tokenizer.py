import torch
from os import truncate
from transformers import LEDTokenizer


def get_tokenizer():    
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    return tokenizer


def tokenize(tokenizer, batch):
    batch = tokenizer(batch, max_length=2000, padding=True, truncation=True, return_tensors='pt')
    return batch


def response_tokenize(tokenizer, batch):
    batch = tokenizer(batch, max_length=300, padding=True, truncation=True, return_tensors='pt')
    return batch