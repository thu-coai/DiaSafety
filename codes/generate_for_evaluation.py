import csv
from tqdm import tqdm
import json
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse

parser = argparse.ArgumentParser(description='choose dataset')
parser.add_argument('--topk', type = int)
parser.add_argument('--topp', type= float)
parser.add_argument('--size', choices=['small','medium','large'])
args = parser.parse_args()

print(args)


# This is an example for generating the responses of models.
# After generating, we can use our classifier to check the safety (including utterance-level and context-sensitive)

# The parameters in this script is easy to change and see what changes in the final results.
mname = 'DialoGPT/{}'.format(args.size)
model = GPT2LMHeadModel.from_pretrained(mname)

model = model.cuda()
tokenizer = GPT2Tokenizer.from_pretrained(mname)

for dataset in ['val','test']:
    with open('diasafety_{}.json'.format(dataset), 'r') as f:
        data = json.load(f)
    sessions = []
    for d in tqdm(data):
        tmp = d['context']
        UTTERANCE = tmp+tokenizer.eos_token
        inputs = tokenizer([UTTERANCE]*10, return_tensors='pt') # for each context, we generate (sample) response for 10 times.

        inputs = inputs.to('cuda')
        reply_ids = model.generate(**inputs,  max_length=128, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=args.topk, top_p=args.topp)
        reply = tokenizer.batch_decode(reply_ids[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        sessions.append({'context': d['context'], 'gen_response': reply})

    with open('/dialo_{}_{}_{}_{}.json'.format(dataset, args.size, args.topk, args.topp),'w') as f:
        json.dump(sessions, f, indent=0)