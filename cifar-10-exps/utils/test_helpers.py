import numpy as np
import torch
import torch.nn as nn

from models.ResNet import ResNetCifar as ResNet


def build_model(args):
    print('Building model...')
    def gn_helper(planes):
        return nn.GroupNorm(args.group_norm, planes)
    net = ResNet(26, 1, channels=3, classes=10, norm_layer=gn_helper).cuda()

    if hasattr(args, 'parallel') and args.parallel:
        net = torch.nn.DataParallel(net)
    return net


def test(dataloader, model):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()
    model.train()
    return 1-correct.mean(), correct, losses
