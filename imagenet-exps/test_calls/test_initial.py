from __future__ import print_function
import argparse

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.test_helpers import test
from utils.train_helpers import build_model, prepare_test_data


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True)
parser.add_argument('--use_rvt', action='store_true')
parser.add_argument('--use_resnext', action='store_true')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--resume', required=True)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)
args = parser.parse_args()

net = build_model(args)
teset, teloader = prepare_test_data(args)

print(f'Resuming from {args.resume}...')
ckpt = torch.load(f'{args.resume}/ckpt.pth')
net.load_state_dict(ckpt['state_dict'])
cls_initial, _, _ = test(teloader, net, args.corruption, verbose=True)

print(f'Error on corrupted test set {cls_initial*100:.2f}')
