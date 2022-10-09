
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import pandas as pd
import numpy as np
from dataset import get_loader, get_test_loader, process_text
from model import get_model
from optimizer import get_optimizer
from tokenizer import get_tokenizer, tokenize, response_tokenize
from tqdm import tqdm
from generate import generate
from scipy.stats import spearmanr, kendalltau, pearsonr

torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device('cuda') 

def cal_cor(data1, data2):
    coef, p = spearmanr(data1, data2)
    print('Spearmans correlation coefficient: %.3f' % coef)
    # interpret the significance
    alpha = 0.01
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)

    coef, p = pearsonr(data1, data2)
    print('Pearsonr correlation coefficient: %.3f' % coef)
    # interpret the significance
    alpha = 0.01
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)


    coef, p = kendalltau(data1, data2)
    print('Kendall correlation coefficient: %.3f' % coef)
    # interpret the significance
    alpha = 0.01
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])



def test(model, tokenizer, test_dataset):
    print('Testing...')
    model.eval()
    correct = 0
    total = 0
    distance = []
    for j, (high_story, low_story, targets, margins, prompt_rank, prompt_rate_reason, aspect_story, story, comments, aspects, aspect_scores) in enumerate(tqdm(test_dataset)):
        targets = targets.to(device)
        margins = margins.to(device)
        high_story = tokenize(tokenizer, high_story)
        low_story = tokenize(tokenizer, low_story)
        outputs1, _, _, _ = model(input_ids=high_story['input_ids'].to(device), attention_masks=high_story['attention_mask'].to(device))
        outputs2, _, _, _ = model(input_ids=low_story['input_ids'].to(device), attention_masks=low_story['attention_mask'].to(device))
        # Acc
        for idx, margin in enumerate(margins):
            if targets[idx].item() == 1:
                if outputs1[idx] > outputs2[idx]:
                    correct += 1
                total += 1
            if targets[idx].item() == -1:
                if outputs2[idx] > outputs1[idx]:
                    correct += 1      
                total += 1
        # Dis
        outputs1 = outputs1.mean().item()
        outputs2 = outputs2.mean().item()
        distance.append(outputs1-outputs2)

    print('Acc:', correct/total*100,'%')
    print('Dis:', np.mean(distance))

    # correlation with human evaluation on WP_200
    print('=================writing prompt===================')
    with open('data/story_correlation.pkl', 'rb') as f:
        story = pickle.load(f)
    data1 = []
    data2 = []
    for story, score in story:
        inputs = tokenize(tokenizer, [story])
        outputs, _, _, _ = model(input_ids=inputs['input_ids'].to(device), attention_masks=inputs['attention_mask'].to(device))
        data1.append(outputs[0].item())
        data2.append(score)
    cal_cor(data1, data2)

    # correlation with human evaluation on SCARY_200
    print('=================scary story===================')
    with open('data/scary_story_correlation.pkl', 'rb') as f:
        scary_story = pickle.load(f)
    data1 = []
    data2 = []
    for story, score in scary_story:
        inputs = tokenize(tokenizer, [story])
        outputs, _, _, _ = model(input_ids=inputs['input_ids'].to(device), attention_masks=inputs['attention_mask'].to(device))
        data1.append(outputs[0].item())
        data2.append(score)
    cal_cor(data1, data2)



def generate_one(model, input_sentence, tokenizer):
    exmaple = tokenize(tokenizer, input_sentence)
    res = generate(model, exmaple, tokenizer)   
    return res

def train_model(model, dataset, optimizer, lr_scheduler, tokenizer, epochs, test_dataset):
    # model.eval()
    # with torch.no_grad():
    #     test(model, tokenizer, test_dataset) 
    model.train()
    aspect_score_criterion = torch.nn.BCELoss()
    aspect_conf_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs): 
        model.train()
        total_dis_loss_list = [] 
        total_gen_loss_list = [] 
        total_a_conf_loss_list = []
        total_a_score_loss_list = []
        cnt = 0
        total = 0
        for j, (high_story, low_story, targets, margins, prompt_rank, prompt_rate_reason, aspect_story, story, comments, aspects, aspect_scores) in enumerate(tqdm(dataset)):
            total_loss = 0
            optimizer.zero_grad()   
            targets = targets.to(device)
            margins = margins.to(device)
            high_story = tokenize(tokenizer, high_story)
            low_story = tokenize(tokenizer, low_story)
            story = tokenize(tokenizer, story)
            aspect_story = tokenize(tokenizer, aspect_story)
            aspects = aspects.to(device)
            aspect_scores = aspect_scores.to(device).float()

            score1, _, _, _ = model(input_ids=high_story['input_ids'].to(device), attention_masks=high_story['attention_mask'].to(device))
            score2, _, _, _ = model(input_ids=low_story['input_ids'].to(device), attention_masks=low_story['attention_mask'].to(device))

            comments = response_tokenize(tokenizer, comments)

            # overall score prediction loss
            overall_score_loss = 0
            for i in range(len(score1)):
                overall_score_loss += max(0, -1*targets[i]*(score1[i]-score2[i])+0.3)
                
            # comment generation loss
            _, _, _, gen_loss = model(input_ids=aspect_story['input_ids'].to(device), attention_masks=aspect_story['attention_mask'].to(device), trg_input_ids=comments['input_ids'].to(device))
            gen_loss = gen_loss.mean()

            # aspect confidence and rating loss
            _, a_conf, a_score, _ = model(input_ids=story['input_ids'].to(device), attention_masks=story['attention_mask'].to(device), trg_input_ids=comments['input_ids'].to(device))
            a_conf_loss = aspect_conf_criterion(a_conf, aspects)

            aspects = aspects.unsqueeze(-1)
            a_score = torch.gather(a_score, 1, aspects)
            a_score_loss = aspect_score_criterion(a_score, aspect_scores.unsqueeze(-1))
            loss = overall_score_loss + gen_loss + a_conf_loss + a_score_loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_dis_loss_list.append(overall_score_loss)
            total_gen_loss_list.append(gen_loss)
            total_a_conf_loss_list.append(a_conf_loss)
            total_a_score_loss_list.append(a_score_loss)

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                test(model, tokenizer, test_dataset) 
            print(f"Epoch {epoch} ce loss, ", "dis: ", (sum(total_dis_loss_list)/len(total_dis_loss_list)).item(), "gen:", (sum(total_gen_loss_list)/len(total_gen_loss_list)).item(), "a_conf:", (sum(total_a_conf_loss_list)/len(total_a_conf_loss_list)).item(), "a_score:", (sum(total_a_score_loss_list)/len(total_a_score_loss_list)).item())
            model.train()
            torch.save(model.module.state_dict(), f'model/{epoch}.pt')


def main():
    # get data loader
    # batch_size = 16
    batch_size = 1 # set to 16 in real experiment
    dataset = get_loader(batch_size)
    test_dataset = get_test_loader(batch_size)
    # get model
    model = get_model()
    model = model.to(device)
    model = nn.DataParallel(model)
    # get optimizer
    LR = 4e-5
    ADAM_EPSILON = 1e-8
    WEIGHT_DECAY = 0.
    WARMUP_PROPORTION = 0.2

    EPOCH = 5
    TRAIN_STEP = EPOCH * (len(dataset) + 1)
    WARMUP_STEP = TRAIN_STEP*WARMUP_PROPORTION

    optimizer, lr_scheduler = get_optimizer(model=model, lr=LR, train_steps=TRAIN_STEP, warmup_steps=WARMUP_STEP, weight_decay=WEIGHT_DECAY, adam_epsilon=ADAM_EPSILON)
    tokenizer = get_tokenizer()

    train_model(model, dataset, optimizer, lr_scheduler, tokenizer, EPOCH, test_dataset)


if __name__ == "__main__":
    main()




