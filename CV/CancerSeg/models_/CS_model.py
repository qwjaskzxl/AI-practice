import sys
import numpy as np, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_.unet import Unet

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
        self.unet = Unet(64)
        # k = 9
        # self.cnn1 = nn.Conv2d(4, 16, kernel_size=(k, k), stride=(1, 1), padding=(k // 2, k // 2), bias=False)
        # self.cnn2 = nn.Conv2d(16, 32, kernel_size=(k, k), stride=(1, 1), padding=(k // 2, k // 2), bias=False)
        # self.cnn3 = nn.Conv2d(32, 16, kernel_size=(k, k), stride=(1, 1), padding=(k // 2, k // 2), bias=False)
        # self.cnn4 = nn.Conv2d(16, 4, kernel_size=(k, k), stride=(1, 1), padding=(k // 2, k // 2), bias=False)

        self.fc = nn.Linear(64, 4)
        self.drop = nn.Dropout2d(p=0.3)
        self.m = nn.Sigmoid()

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Linear(d, args.cls_num)
        self.CE = nn.CrossEntropyLoss(reduction='mean')
        self.BCE = nn.BCELoss()

    def stack_cnn(self):
        pass

    def forward(self, img, y, **batch):
        loss_cls = torch.Tensor([0]).to(img.device)

        # B, C, H, W = img.size()
        # x = img
        # x = self.cnn1(x)
        # x = self.cnn2(x)
        # x = self.cnn3(x)
        # x = self.cnn4(x)
        # loss_l1 = torch.abs(x - y).mean()

        feat_map = self.unet(img)
        feat_map = self.fc(feat_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        loss_l1 = torch.abs(feat_map - y).mean()

        # # logits_map = self.m(feat_map.sum(1, keepdim=True))
        # # loss_gen = self.BCE(logits_map, y_mask)
        # # x = self.maxpool(feat_map).view(B, -1)
        # # x = self.drop(x)
        # # prob = self.cls(x)
        # # loss_cls = self.CE(prob, label)

        loss = loss_l1  # + loss_cls
        return {
            'feat_map': feat_map,
            # 'prob': prob,
            'loss': loss,
            'loss_l1': loss_l1,
        }


if __name__ == '__main__':
    pass
