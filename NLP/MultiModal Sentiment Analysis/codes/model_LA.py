import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fc import MLP, FC
from layers.layer_norm import LayerNorm


# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


def mask_mask2(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0)


# ------------------------------
# ---------- Flattening --------
# ------------------------------


class AttFlat(nn.Module):
    def __init__(self, args, flat_glimpse, merge=False):
        super(AttFlat, self).__init__()
        self.args = args
        self.merge = merge
        self.flat_glimpse = flat_glimpse
        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        if self.merge:
            self.linear_merge = nn.Linear(
                args.hidden_size * flat_glimpse,
                args.hidden_size * 2
            )

    def forward(self, x, x_mask):
        att = self.mlp(x)  # 80*60*512 -》 80*60*60
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)

            return x_atted

        return torch.stack(att_list).transpose_(0, 1)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()
        # 多头注意力机制
        self.mhatt = MHAtt(args)
        # 前馈神经网络
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, args):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):  # x：音频，y：文本
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ---------------------------
# ---- FF + norm  -----------
# ---------------------------
# class FFAndNorm(nn.Module):
#     def __init__(self, args):
#         super(FFAndNorm, self).__init__()
#
#         self.ffn = FFN(args)
#         self.norm1 = LayerNorm(args.hidden_size)
#         self.dropout2 = nn.Dropout(args.dropout_r)
#         self.norm2 = LayerNorm(args.hidden_size)
#
#     def forward(self, x):
#         x = self.norm1(x)
#         x = self.norm2(x + self.dropout2(self.ffn(x)))
#         return x


class Block(nn.Module):
    def __init__(self, args, i):
        super(Block, self).__init__()
        self.args = args
        # 子注意力机制Self Attention
        self.sa1 = SA(args)
        #  Self Guided Attention
        self.sa3 = SGA(args)

        # self.last = (i == args.layer)
        # if not self.last:
        self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
        self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
        self.norm_l = LayerNorm(args.hidden_size)
        self.norm_i = LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask):
        # ax, ay：80*60*512
        # x_mask, y_mask:80*1*1*60, x,y :80*60*512
        ax = self.sa1(x, x_mask)
        ay = self.sa3(y, x, y_mask, x_mask)

        x = ax + x
        y = ay + y

        # if self.last:
        #  return x, y

        ax = self.att_lang(x, x_mask)
        ay = self.att_audio(y, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_i(y + self.dropout(ay))


class Model_LA(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb):
        super(Model_LA, self).__init__()

        self.args = args

        # LSTM
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.word_embed_size
        )
        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.lstm_x = nn.LSTM(
            input_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # self.lstm_y = nn.LSTM(
        #     input_size=args.audio_feat_size,
        #     hidden_size=args.hidden_size,
        #     num_layers=1,
        #     batch_first=True
        # )

        # Feature size to hid size
        self.adapter = nn.Linear(args.audio_feat_size, args.hidden_size)

        # Encoder blocks
        self.enc_list = nn.ModuleList([Block(args, i) for i in range(args.layer)])

        # Flattenting features before proj
        # self.attflat_img = AttFlat(args, 1, merge=True)
        self.attflat_lang = AttFlat(args, 1, merge=True)

        # attention
        #   self.sa1 = SA(args)
        self.mhat = MHAtt(args)
        self.proj_norm_sa = LayerNorm(args.hidden_size)
        # Classification layers
        self.proj_norm = LayerNorm(2 * args.hidden_size)
        self.proj = self.proj = nn.Linear(2 * args.hidden_size, args.ans_size)
        self.dropoutmhat = nn.Dropout(args.dropout_r)
        self.normmhat = LayerNorm(args.hidden_size)
        # 文本token权重
        self.register_parameter(name="w1", param=torch.nn.Parameter(
            torch.randn(args.audio_seq_len, requires_grad=True).reshape(args.audio_seq_len, 1)))

        # 语音token权重
        self.register_parameter(name="w2", param=torch.nn.Parameter(
            torch.randn(args.audio_seq_len, requires_grad=True).reshape(args.audio_seq_len, 1)))

    # 权重矩阵乘积
    def wx(self, x, w):
        return torch.mul(x, w)

    def forward(self, x, y):
        args = self.args
        args.mode = 'LA'
        if 'L' in args.mode:
            # x_mask 80*1*1*60
            x_mask = make_mask(x.unsqueeze(2))
            # 80*60*300 编码
            embedding = self.embedding(x)
            # 80*60*512 经过lstm编码
            x, _ = self.lstm_x(embedding)
            proj_feat = self.wx(x, self.w1)
        if 'A' in args.mode:
            # y_mask 80*1*1*60
            y_mask = make_mask(y)
            # y, _ = self.lstm_y(y)
            # Feature size to hid size  *80*60*512
            y = self.adapter(y)
            proj_feat = self.wx(y, self.w2)
        if 'LA' in args.mode:
            for i, dec in enumerate(self.enc_list):
                x_m, x_y = None, None
                if i == 0:
                    x_m, x_y = x_mask, y_mask
                x, y = dec(x, x_m, y, x_y)
            proj_feat = self.wx(x, self.w1) + self.wx(y, self.w2)
        # # x 80*60*512->80*1024
        # x = self.attflat_lang(
        #     x,
        #     None
        # )
        # # y 80*60*512->80*1024
        # y = self.attflat_img(
        #     y,
        #     None
        # )

        # Classification layers

        # 向量相加后的标准化
        proj_feat = self.proj_norm_sa(proj_feat)
        # attention
        proj_mask = make_mask(proj_feat)
        proj_feat = self.normmhat(self.dropoutmhat(self.mhat(proj_feat, proj_feat, proj_feat, proj_mask)))
        #  flat 80*60*512 -> 80*1024
        proj_feat = self.attflat_lang(proj_feat, None)
        proj_feat = self.proj_norm(proj_feat)
        ans = self.proj(proj_feat)
        # 80*7 or 6 or 3
        return ans
