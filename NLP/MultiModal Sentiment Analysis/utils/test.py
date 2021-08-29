#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @Time：2021/2/269:26
# @Author:sam
# @File:tt
import torch
import torch.nn.functional as F

# 2 * 3 * 4
t = torch.FloatTensor([[[1, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]])
# 2 * 3 * 3
a = torch.FloatTensor([[[1, 2, 3], [4, 5, 8], [4, 5, 8]], [[7, 8, 9], [10, 11, 12], [10, 11, 12]]])

tmp_list = []
b = F.softmax(a,dim=1)
print(b)
print(t)

for i in range(2):
    dd = b[:, :, i:i+1]    #  2 * 3 * 1
    print(dd)
    tmp = b[:, :, i:i+1] * t
    print(tmp)#2*3*4
    sum = torch.sum(tmp,dim=1)#2*4
    print(sum)
    tmp_list.append(sum)
    print(tmp_list)

