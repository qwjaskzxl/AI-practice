import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from a3_graph.pygcn.utils import load_data, accuracy
from a3_graph.pygcn.models import GCN
from a3_graph.pygcn.config import get_args

from tensorboardX import SummaryWriter
writer = SummaryWriter('../tensorboard')

def train(epoch, model, optimizer):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    # print(features)
    # print(adj)
    writer.add_graph(model, input_to_model=[features, adj.to_dense()], verbose=False)
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    writer.add_histogram('loss_train', loss_train.item(), epoch)
    writer.add_histogram('acc_train', acc_train.item(), epoch)
    writer.add_histogram('loss_test', loss_val.item(), epoch)
    writer.add_histogram('acc_test', acc_val.item(), epoch)  # add_scalar


def test(model):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
    # import os
    # print(os.getcwd())
    # 获取config
    args = get_args()

    # 加载数据
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # 模型与优化器定义
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # 转移到cuda
    if not args.no_cuda and torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        torch.cuda.manual_seed(args.seed)

    #固定随机种子，保证模型可复现性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 模型训练
    start = time.time()
    for epoch in range(args.epochs):
        train(epoch, model, optimizer)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start))

    # 测试
    test(model)