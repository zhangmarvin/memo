from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tqdm import tqdm
from utils.adapt_helpers import adapt_single, test_single
from utils.train_helpers import build_model, prepare_test_data


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True)
parser.add_argument('--use_rvt', action='store_true')
parser.add_argument('--use_resnext', action='store_true')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--resume', required=True)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--prior_strength', default=16, type=int)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--lr', default=0.00025, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--niter', default=1, type=int)
args = parser.parse_args()

net = build_model(args)

teset, _ = prepare_test_data(args, use_transforms=False)

print(f'Resuming from {args.resume}...')
ckpt = torch.load('{args.resume}/ckpt.pth')

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

print('Running...')
correct = []
for i in tqdm(range(len(teset))):
    net.load_state_dict(ckpt['state_dict'])
    image, label = teset[i-1]
    adapt_single(net, image, optimizer, marginal_entropy,
                 args.corruption, args.niter, args.batch_size, args.prior_strength)
    correct.append(test_single(net, image, label, args.corruption, args.prior_strength)[0])

print('MEMO adapt test error {(1-np.mean(correct))*100:.2f}')
