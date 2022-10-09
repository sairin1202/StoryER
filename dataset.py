import torch
import os
import pickle
import json
import random
import pickle
import ast
import re
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from tokenizer import tokenize, get_tokenizer
from copy import deepcopy
from tqdm import tqdm
from glob import glob

def remove_space(text):
    return re.sub(' +', ' ', text)

def replace_newline(text):
    text = text.replace('<\\newline> <\\newline>', '<\\newline>')
    text = text.replace('<\\newline> <\\newline> <\\newline>', '<\\newline>')
    text = text.replace('<\\newline> <\\newline> <\\newline> <\\newline>', '<\\newline>')
    return text

# replace some tokens
def process_text(text):
    text = text.replace('\t', ' ')
    text = text.replace('\n', ' <\\newline> ')
    text = text.replace('<br>', ' <\\newline> ')
    text = remove_space(text)
    text = replace_newline(text)
    return text



class Dataset():
    def __init__(self):

        if os.path.exists('data/train_rank_data_small.pkl'):
            with open('data/train_rank_data_small.pkl', 'rb') as f:            
                self.train_rank_data = pickle.load(f)

        # with open('data/aug_comment.pkl', 'rb') as f:
        #     aug_rate_reason_data = pickle.load(f)
            
        with open('data/train_rate_reason_data_small.json', 'r') as f:
            self.train_rate_reason_data = json.load(f)
            self.train_rate_reason_data = [(d['prompt'], d['story'], a-1, c, (r-1)/4) for d in self.train_rate_reason_data for (a, r, c) in d['comments'] if type(a)== int and type(c)!=float and a != -1]
            # self.train_rate_reason_data.extend(aug_rate_reason_data)

        self.aspect_string = ['opening/beginning', 'middle/twist/flow', 'ending', 'character shaping', 'scene description', 'heartwarming/touch', 'sad/crying/tear', 'horror/scary', 'funny/hilarious/laugh', 'novel/idea']

    def __getitem__(self, index):
        prompt_rank, high_story, low_story, target, margin = self.train_rank_data[index]
        index_gen = random.randint(0, len(self.train_rate_reason_data)-1)
        prompt_rate_reason, aspect_story, aspect, comment, aspect_rate = self.train_rate_reason_data[index_gen]
        aspect_story = process_text(aspect_story)
        return high_story, low_story, target, margin, prompt_rank, prompt_rate_reason, self.aspect_string[aspect] + ' <sep> ' + aspect_story, aspect_story, comment, aspect, aspect_rate

    def __len__(self):
        return len(self.train_rank_data)



def get_loader(batch_size, shuffle=True):
    dataset = Dataset()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=10,
                                              drop_last=True)
    return data_loader




class TestDataset():
    def __init__(self):

        if os.path.exists('data/test_rank_data.pkl'):
            with open('data/test_rank_data.pkl', 'rb') as f:            
                self.test_rank_data = pickle.load(f)
            
        with open('data/test_rate_reason_data.json', 'r') as f:
            self.test_rate_reason_data = json.load(f)
            self.test_rate_reason_data = [(d['prompt'], d['story'], a-1, c, (r-1)/4) for d in self.test_rate_reason_data for (a, r, c) in d['comments'] if type(a)== int and type(c)!=float and a != -1]
        
        self.aspect_string = ['opening/beginning', 'middle/twist/flow', 'ending', 'character shaping', 'scene description', 'heartwarming/touch', 'sad/crying/tear', 'horror/scary', 'funny/hilarious/laugh', 'novel/idea']

    def __getitem__(self, index):
        prompt_rank, high_story, low_story, target, margin = self.test_rank_data[index]
        index_gen = random.randint(0, len(self.test_rate_reason_data)-1)
        prompt_rate_reason, aspect_story, aspect, comment, aspect_rate = self.test_rate_reason_data[index_gen]
        # print(aspect)
        return high_story, low_story, target, margin, prompt_rank, prompt_rate_reason, self.aspect_string[aspect] + ' <sep> ' + aspect_story, aspect_story, comment, aspect, aspect_rate

    def __len__(self):
        return len(self.test_rank_data)


def get_test_loader(batch_size, shuffle=True):
    dataset = TestDataset()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=10,
                                              drop_last=True)
    return data_loader
