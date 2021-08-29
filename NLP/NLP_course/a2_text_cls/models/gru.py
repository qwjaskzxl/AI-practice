import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, n_gram=1, bidirectional=False):
        super(GRU, self).__init__()
        # 完美复现
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.hidden_size = hidden_size

        self.Wz = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Uz = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wr = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Ur = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Uh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # self.Wz = nn.Linear(input_size, hidden_size, False) #全加bias会下降0.2%
        # self.Uz = nn.Linear(hidden_size, hidden_size, False)
        # self.Wr = nn.Linear(input_size, hidden_size, False)
        # self.Ur = nn.Linear(hidden_size, hidden_size, False)
        # self.Wh = nn.Linear(input_size, hidden_size, False)
        # self.Uh = nn.Linear(hidden_size, hidden_size, False)
        self.reset_parameters()

    def reset_parameters(self):
        # 这样也是完美复现
        for weight in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]:
            init.kaiming_uniform_(weight, a=math.sqrt(5))  # 均匀分布
            # init.kaiming_normal_(weight) #正态分布
            # init.xavier_uniform_(weight)
            # init.xavier_normal_(weight)

        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b,seq_len,emb_dim], [b, 32, 300] here
        bs, seq_len, emb_dim = x.shape
        Ht = torch.zeros(bs, seq_len, self.hidden_size).cuda()
        ht = torch.randn(bs, self.hidden_size).cuda()
        # init.kaiming_uniform_(ht, a=math.sqrt(5)) #last变好，sum变差

        for t in range(seq_len):
            x_ = x[:, t, :]  # [bs,emb_dim]
            # zt = self.sig(self.Wz(x_) + self.Uz(ht))
            # rt = self.sig(self.Wr(x_) + self.Ur(ht))  # [b,h]
            # ht_wave = self.tanh(self.Wh(x_) + self.Uh(rt * ht))
            zt = self.sig(torch.matmul(x_, self.Wz) + torch.matmul(ht, self.Uz))
            rt = self.sig(torch.matmul(x_, self.Wr) + torch.matmul(ht, self.Ur))
            ht_wave = self.tanh(torch.matmul(x_, self.Wh) + torch.matmul(rt * ht, self.Uh))
            ht = (1 - zt) * ht + zt * ht_wave
            Ht[:, t, :] = ht
        return Ht, None


class GRU_ngram_attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, n_gram=3, n_head=2, attn_type=1, bidirectional=True):
        super(GRU_ngram_attn, self).__init__()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_g = n_gram
        self.n_h = n_head
        self.attn_type = attn_type
        self.bidirectional = bidirectional

        self.Wz = [nn.Linear(input_size, hidden_size, False).cuda() for i in range(2)]
        self.Uz = [nn.Linear(hidden_size, hidden_size, False).cuda() for i in range(2)]
        self.Wr = [nn.Linear(input_size, hidden_size, False).cuda() for i in range(2)]
        self.Ur = [nn.Linear(hidden_size, hidden_size, False).cuda() for i in range(2)]
        self.Wh = [nn.Linear(input_size, hidden_size, False).cuda() for i in range(2)]
        self.Uh = [nn.Linear(hidden_size, hidden_size, False).cuda() for i in range(2)]
        # 按照公式里来的
        self.Ua1 = [[nn.Linear(input_size, 1, False).cuda() for i in range(n_head)] for j in range(2)]
        self.Wa1 = [[nn.Linear(hidden_size, n_gram, True).cuda() for i in range(n_head)] for j in range(2)]
        # self.Wa2 = [[nn.Linear(input_size, hidden_size, False).cuda() for i in range(n_head)] for j in range(2)]
        self.Wa2 = [[nn.Bilinear(input_size, hidden_size, 1, bias=False).cuda() for i in range(n_head)] for j in range(2)]
        self.fusion_head = [nn.Linear(n_head, 1).cuda() for i in range(2)]
        self.fusion_bidir = nn.Linear(60, 30)


    def forward(self, x):
        if not self.bidirectional:
            return self.forward_dir(x, 0)
        else:
            X_hat0 = self.forward_dir(x, 0)
            # x = torch.from_numpy(x.cpu().detach().numpy()[:, ::-1, :])
            x1 = torch.zeros(x.shape).cuda()
            for i in range(x.shape[1]):
                x1[:,i,:] = x[:,x.shape[1]-i-1,:]
            X_hat1 = self.forward_dir(x1, 1)
            X_bidir = torch.cat([X_hat0, X_hat1], dim=1).permute(0,2,1)
            return self.fusion_bidir(X_bidir).permute(0,2,1)


    def forward_dir(self, x, d):
        # x:[b,seq_len,emb_dim], [b, 32, 300] here
        bs, seq_len, emb_dim = x.shape
        mid_len = seq_len - self.n_g + 1  # n_grzam之后中间的长度

        ht = torch.randn(bs, self.hidden_size).cuda()
        Ht = torch.zeros(bs, seq_len, self.hidden_size).cuda()
        X_hat = torch.zeros(bs, mid_len, self.n_g, self.input_size).cuda()

        for t in range(mid_len):
            x_ = x[:, t:t + self.n_g, :]  # [bs, n_g, emb_dim]
            if self.attn_type == 1:
                et_head = [self.Ua1[d][i](x_) + self.Wa1[d][i](ht).unsqueeze(2) for i in range(self.n_h)]
            elif self.attn_type == 2:
                # et_head = [(self.Wa2[d][i](x_) * ht.unsqueeze(1)).sum(2).unsqueeze(2) for i in range(self.n_h)]
                et_head = [(self.Wa2[d][i](x_.reshape(-1,x_.shape[-1]),
                                           ht.unsqueeze(1).repeat(1,self.n_g,1).reshape(-1,ht.shape[-1]))).reshape(bs,self.n_g,1)
                           for i in range(self.n_h)]

            et_head = torch.cat(et_head, dim=2)
            et = self.tanh(self.fusion_head[d](et_head).squeeze())  # [bs, n_g]
            alpha = F.softmax(et, dim=1)
            X_hat[:, t, :] = x_ * alpha.unsqueeze(2)

            x_ = x_.sum(1)
            zt = self.sig(self.Wz[d](x_) + self.Uz[d](ht))
            rt = self.sig(self.Wr[d](x_) + self.Ur[d](ht))  # [b,h]

            ht_wave = self.tanh(self.Wh[d](x_) + self.Uh[d](rt * ht))
            ht = (1 - zt) * ht + zt * ht_wave
            Ht[:, t, :] = ht

        return X_hat.sum(2)