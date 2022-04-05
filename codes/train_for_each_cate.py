from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch
import json
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from time import time
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import torch.nn as nn
import random
import os

def get_loader(dataset, tokenizer, batchsize=16, padsize=256, want_label=1):

    batch_inputs, batch_labels = [], []
    
    inputs1, inputs2, labels_ = [d['context'] for d in dataset], [d['response'] for d in dataset], [d['label'] for d in dataset]
    labels = []
    for label in labels_:
        if label==want_label:
            labels.append(1)
        elif label==want_label-1:
            labels.append(0)
        else:
            labels.append(2)
        #labels.append(int(label==want_label))
    for start in tqdm(range(0, len(inputs1), batchsize)):
        tmp_batch = tokenizer(text=inputs1[start:min(start + batchsize, len(inputs1))],
                              text_pair=inputs2[start:min(start + batchsize, len(inputs1))],
                              return_tensors="pt", truncation=True, padding='max_length', max_length=padsize)
        batch_inputs.append(tmp_batch)
        tmp_label = torch.LongTensor(labels[start:min(start + batchsize, len(inputs1))])
        batch_labels.append(tmp_label)
    return batch_inputs, batch_labels


def get_loader_resp(dataset, tokenizer, batchsize=16, padsize=256, want_label=1):
    batch_inputs, batch_labels = [], []
    inputs1, inputs2, labels_ = [d['context'] for d in dataset], [d['response'] for d in dataset], [d['label'] for d in dataset]
    labels = []
    for label in labels_:
        if label==want_label:
            labels.append(1)
        elif label==want_label-1:
            labels.append(0)
        else:
            labels.append(2)
    for start in tqdm(range(0, len(inputs2), batchsize)):
        tmp_batch = tokenizer(text=inputs2[start:min(start + batchsize, len(inputs2))],
                              return_tensors="pt", truncation=True, padding='max_length', max_length=padsize)
        batch_inputs.append(tmp_batch)
        tmp_label = torch.LongTensor(labels[start:min(start + batchsize, len(inputs2))])
        batch_labels.append(tmp_label)
    return batch_inputs, batch_labels


def evaluate(model, batch_inputs, batch_labels,test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for inputs, labels in zip(batch_inputs, batch_labels):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            loss = loss_fct(logits, labels)
            loss_total += loss

            labels = labels.view(-1).data.cpu().numpy()
            predic = torch.max(logits.view(-1, logits.shape[-1]).data, 1)[1].cpu()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(batch_inputs), report, confusion, labels_all, predict_all
    return acc, loss_total / len(batch_inputs), f1


def test_report(model, save_path, batch_inputs, batch_labels, log_file):
    # test
    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time()
    test_acc, test_loss, test_report, test_confusion, label, predict = evaluate(model, batch_inputs, batch_labels,
                                                                                test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc), file=log_file)
    print("Precision, Recall and F1-Score...")
    print(test_report, file=log_file)
    print("Confusion Matrix...")
    print(test_confusion, file=log_file)
    time_dif = time() - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print("Time usage:", time_dif, file=log_file)

parser = argparse.ArgumentParser(description='choose dataset')
parser.add_argument('dataset', choices=['agreement', 'expertise', 'offend','political','bias','risk'])
args = parser.parse_args()

with open('diasafety_train.json', 'r') as f:
    train = json.load(f)

with open('diasafety_val.json', 'r') as f:
    val = json.load(f)

with open('diasafety_test.json', 'r') as f:
    test = json.load(f)

label_dict = {'agreement':1, 'expertise':3, 'offend':5, 'political':7, 'bias':9, 'risk':11} # political class is finally deprecated

want_label = label_dict[args.dataset]

num_labels = 3 # (safe, unsafe, N/A) for a specific category

padsize = 128
num_epochs = 10

require_improvement = 2000 # can be adjusted


import itertools
batchsizes = [64, 32, 16, 8, 4]

learning_rates = [5e-3,2e-3,5e-4,2e-4,5e-5,2e-5,5e-6,2e-6]

weight = [1,1,1] # can be adjuested
weight = torch.FloatTensor(weight)
import sys
#log_file = sys.stdout
for batchsize, learning_rate in itertools.product(batchsizes,learning_rates):
    path = 'roberta-base'
    if not os.path.isdir('../models_{}'.format(args.dataset)):
        os.mkdir('../models_{}'.format(args.dataset))
    save_path = '../models_{}/model_{}_{}'.format(args.dataset, batchsize, learning_rate)
    tokenizer = RobertaTokenizer.from_pretrained(path)
    model = RobertaForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    total_batch = 0
    dev_best_loss = float('inf')
    best_f1 = 0
    last_improve = 0
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("getting loader...")
    val_inputs, val_labels = get_loader(val, tokenizer, batchsize=batchsize, padsize=padsize,want_label=want_label)
    test_inputs, test_labels = get_loader(test, tokenizer, batchsize=batchsize, padsize=padsize, want_label=want_label)


    model = model.to(device)
    flag = False
    weight = weight.to(device)
    loss_fct = nn.CrossEntropyLoss(weight=weight)
    print("start to train...")
    for epoch in range(num_epochs):
        model.train()
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        start_time = time()
        random.seed(42)
        random.shuffle(train)
        train_inputs, train_labels = get_loader(train, tokenizer, batchsize=batchsize, padsize=padsize, want_label=want_label)
        for i, (trains, labels) in enumerate(zip(train_inputs, train_labels)):
            trains, labels = trains.to(device), labels.to(device)
            outputs = model(**trains, labels=labels)

            #loss = outputs.loss
            logits = outputs.logits
            loss = loss_fct(logits, labels)

            model.zero_grad()

            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.view(-1).data.cpu()
                predic = torch.max(logits.view(-1, logits.shape[-1]).data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss, dev_f1 = evaluate(model, val_inputs, val_labels)
                if dev_f1>best_f1:
                    best_f1 = dev_f1
                #if dev_loss < dev_best_loss:
                #    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = time() - start_time
                time_dif = timedelta(seconds=int(round(time_dif)))
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%}  Time: {6} {7}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, dev_f1, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    if not os.path.isdir('../logs_{}'.format(args.dataset)):
        os.mkdir('../logs_{}'.format(args.dataset))
    log_file = open('../logs_{}/log_{}_{}.txt'.format(args.dataset, batchsize, learning_rate),'w')
    print('batchsize: {}\nlearning_rate:{}'.format(batchsize,learning_rate), file=log_file)
    test_report(model, save_path, test_inputs, test_labels, log_file=log_file)
    log_file.close()
