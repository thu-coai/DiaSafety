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
        #labels.append(int(label==want_label))
    for start in tqdm(range(0, len(inputs2), batchsize)):
        tmp_batch = tokenizer(text=inputs2[start:min(start + batchsize, len(inputs2))],
                              return_tensors="pt", truncation=True, padding='max_length', max_length=padsize)
        batch_inputs.append(tmp_batch)
        tmp_label = torch.LongTensor(labels[start:min(start + batchsize, len(inputs1))])
        batch_labels.append(tmp_label)
    return batch_inputs, batch_labels

def predict(model, batch_inputs):
    model.eval()
    probs_all = np.zeros((0,3))
    with torch.no_grad():
        for inputs in tqdm(batch_inputs):
            inputs = inputs.to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all = np.concatenate((probs_all, prob),axis=0)
    return probs_all

label_dict = {'agreement':1, 'expertise':3, 'offend':5, 'bias':9, 'risk':11} # political class is deprecated
with open('diasafety_test.json', 'r') as f:
    test = json.load(f)
import pandas as pd
df = pd.DataFrame.from_dict(test)
batchsize = 256
padsize = 128
device = 'cuda:0'

num_labels = 3
path = 'roberta-base'

save_path_dict = {
            'offend':'models_offend/model',
            'agreement':'models_agreement/model',
            'risk': 'models_risk/model',
            'expertise': 'models_expertise/model',
            'bias':'models_bias/model'
        }

model = RobertaForSequenceClassification.from_pretrained(path, num_labels=num_labels)
tokenizer = RobertaTokenizer.from_pretrained(path)
for dataset in label_dict: 
    print(dataset)
    want_label = label_dict[dataset]
    save_path = save_path_dict[dataset]
    model.load_state_dict(torch.load(save_path))
    model = model.to(device)

    print("getting loader...")
    test_inputs, test_labels= get_loader(test, tokenizer, batchsize=batchsize, padsize=padsize,want_label=want_label)

    print("start to predict...")
    probs = predict(model, test_inputs)
    predicts = np.argmax(probs, axis=1)
    print(probs.shape)
    idx2ann = ['safe', 'ctx_unsafe','others']
    for i in range(3):
        df[dataset+'_'+idx2ann[i]] = list(probs[:,i])
    df[dataset+'_predict']=list(predicts)
    
df.to_csv('test_ensemble.csv')
# saved


# start to ensemble the results

import pandas as pd

df = pd.read_csv('test_ensemble.csv')

label_dict = {'agreement':1, 'expertise':3, 'offend':5, 'bias':7, 'risk':9}
multi_unsafe = []
no_unsafe = []
labels = []
preds = []
for i,row in df.iterrows():
    cnt=0
    tmp_preds = []
    if int(row['label'])%2==0:
        labels.append(0)
    else:
        tmp = label_dict[row['source']]
        labels.append((tmp+1)//2)
    flag=True
    tmp_preds = []
    for dataset in label_dict:
        if int(row[dataset+'_predict'])==1:
            flag=False
            tmp_preds.append(((label_dict[dataset]+1)//2, row[dataset+'_ctx_unsafe']))
    if flag:
        preds.append(0)
    else:
        p = sorted(tmp_preds, key=lambda x:x[1], reverse=True)[0][0] # choose the most confident result



from sklearn.metrics import classification_report, confusion_matrix
report=classification_report(labels, preds, digits=3, target_names=['safe','agreement', 'expertise', 'offend', 'bias', 'risk'])
confusion = confusion_matrix(labels, preds)

print(report)
print(confusion)


