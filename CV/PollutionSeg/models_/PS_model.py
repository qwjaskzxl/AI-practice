import sys
import numpy as np, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_.modules_.transformer import PositionalEncoding, make_transformer_encoder, make_transformer_decoder, Embeddings
from models_.modules_.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from itertools import groupby
from torch import cosine_similarity
from time import time
from math import ceil

sys.path.append('../')


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        d = args.d_model

        self.resnet = resnet18(pretrained=True)
        self.drop = nn.Dropout2d(p=0.3)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Linear(args.d_model, args.cls_num)
        self.CELoss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, img, label=None, **batch):
        loss_cls = torch.Tensor([0]).to(img.device)

        B, C, H, W = img.size()

        x = self.resnet(img)

        x = self.maxpool(x).view(B, -1)
        x = self.drop(x)
        prob = self.cls(x)
        ret = {'prob': prob}

        if batch['train']:
            loss_cls = self.CELoss(prob, label)
            loss = loss_cls
            ret['loss'] = loss
            ret['loss_cls'] = loss_cls
        return ret


if __name__ == '__main__':
    pass
