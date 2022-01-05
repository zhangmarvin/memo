from __future__ import print_function

import argparse

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.prepare_dataset import prepare_test_data
from utils.test_helpers import build_model, test


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True)
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--resume', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=8, type=int)
args = parser.parse_args()

net = build_model(args)
teset, teloader = prepare_test_data(args)

print(f'Resuming from {args.resume}...')
ckpt = torch.load(args.resume + '/ckpt.pth')
net.load_state_dict(ckpt['net'])
cls_initial, _, _ = test(teloader, net)

print(f"Error on original test set {ckpt['err_cls']*100:.2f}")
print(f'Error on corrupted test set {cls_initial*100:.2f}')
