import time

import numpy as np
import torch
import torch.nn as nn

from utils.third_party import AverageMeter, ProgressMeter, imagenet_r_mask, indices_in_1k


def test(teloader, model, corruption, verbose=False, print_freq=10):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(teloader), batch_time, top1, prefix='Test: ')
    one_hot = []
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    end = time.time()

    for i, (inputs, labels) in enumerate(teloader):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            if corruption == 'rendition':
                outputs = outputs[:, imagenet_r_mask]
            elif corruption == 'adversarial':
                outputs = outputs[:, indices_in_1k]
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)
    print(f' * Acc@1 {top1.avg:.3f}')

    if verbose:
        one_hot = torch.cat(one_hot).numpy()
        losses = torch.cat(losses).numpy()
        return 1-top1.avg, one_hot, losses
    else:
        return 1-top1.avg
