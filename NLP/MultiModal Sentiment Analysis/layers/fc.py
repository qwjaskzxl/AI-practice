import torch.nn as nn


class FC(nn.Module):
    """定义全连接层"""
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        # 定义线性层
        self.linear = nn.Linear(in_size, out_size)
        # 是否使用激活函数relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        # 是否使用dropout，减少过拟合
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        # 如果使用激活函数relu，则做relu计算
        if self.use_relu:
            x = self.relu(x)
        # 如果使用dropout，则做dropout计算
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    """定义前馈神经网络2层"""
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        # 第一层全连接层（包含激活函数和dropout），(batch_size, max_len, in_size) -> (batch_size, max_len, mid_size)
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        # 第二层线性层，(batch_size, max_len, mid_size) -> (batch_size, max_len, out_size)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        # 在FC的基础上再增加一个线性层
        return self.linear(self.fc(x))
