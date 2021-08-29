import os, argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
from torch import cosine_similarity
from tqdm import tqdm

# from torchtext import data, datasets
#         from torchtext.vocab import Vectors
#         with open('data/wiki_corpus_tmp.txt', 'r') as f:
#             corpus = f.readlines()[:]
#             TEXT = data.Field(sequential=True, tokenize=self.tokenizer)
#             TEXT.build_vocab(corpus)  # , max_size=30000)
#             print(TEXT.vocab.itos[2])
#             print(TEXT.vocab.vectors)
# def tokenizer(self, text):
#     return text.strip().split(' ')
with open('data/wiki_vocab_min300.json', 'r') as f:  # 200:67381, 300:50742
    vocab = json.load(f)
    print('词数:', len(vocab))
    id2word = {i: w for w, i in vocab.items()}


class W2VDataset(Dataset):
    def __init__(self, vocab, sample_n, max_seq_len, window, neg_rate):
        with open('data/wiki_corpus.txt', 'r') as f:
            corpus = f.readlines()[:]
            print('句数:', len(corpus))

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.window = window
        self.sample_n = sample_n
        self.neg_rate = neg_rate
        self.data = []
        for s in corpus[:20000]:
            s = [vocab[w] for w in s.split() if w in vocab]
            seg = len(s) // max_seq_len
            for i in range(seg):
                self.data.append(s[i * max_seq_len:(i + 1) * max_seq_len])
            if seg > 1:
                self.data.append(s[-max_seq_len:])

    def __getitem__(self, idx):
        s = np.array(self.data[idx])
        c_choice = np.random.choice(range(self.window, self.max_seq_len - self.window), self.sample_n, replace=False)
        center_word = s[c_choice]  # .view(-1, 1).repeat(1, self.neg_rate)
        # print(center_word.shape)
        # exit()
        scope = list(range(-self.window, 0)) + list(range(1, self.window + 1))
        pos_choice = c_choice + np.random.choice(scope, self.sample_n, replace=True)
        pos_word = s[pos_choice]
        # pos_word = np.concatenate([s[c_choice - 1][:, np.newaxis], s[c_choice + 1][:, np.newaxis],
        #                            s[c_choice - 2][:, np.newaxis], s[c_choice + 2][:, np.newaxis]], axis=1)
        global_neg_word = np.random.choice(len(self.vocab), self.sample_n * self.neg_rate, replace=False).reshape(self.sample_n, self.neg_rate)
        # for V in [center_word, pos_word, global_neg_word]:
        #     print(V.shape)
        return center_word, pos_word, global_neg_word

    def __len__(self):
        return len(self.data)


class SkipGramModel(nn.Module):
    def __init__(self, vocab: dict, hiddem_dim, sample_n, neg_rate):
        super().__init__()
        self.sample_n = sample_n
        self.neg_rate = neg_rate
        self.w2v = nn.Parameter(torch.ones(len(vocab), hiddem_dim))
        nn.init.kaiming_normal_(self.w2v)
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, center, pos_word, neg_word):
        b = center.size(0)
        center_vec = self.w2v[center.view(-1)]
        pos_vec = self.w2v[pos_word.view(-1)]
        neg_vec = self.w2v[neg_word.view(-1)]
        # print(center_vec.shape, pos_vec.shape, neg_vec.shape)
        # # 1. 用CE loss，要矩阵乘法。优化方法：
        # (1) 想到一个loss，就是把label的概率分布变成soft的，在window内：把context word分配权重，而不是某个词概率是1：smooth label
        # (2）负采样可以只减少这个矩阵维度就行了
        # (3) hierarchical softmax
        # # loss = self.CELoss(pred, pos_word.view(-1).cuda())
        # pred = torch.matmul(center_vec, self.w2v.permute(1, 0))  # [n,v]
        # pred = pred.softmax(dim=-1)
        # pred = pred[torch.arange(center.size(0)), pos_word.view(-1)]
        # loss = -torch.log(pred).mean()
        # # exit()
        # return loss

        # 2. sigmoid loss，这个可以直接存表，然后查就行了。。
        # pos = torch.sigmoid((center_vec * pos_vec).mean())
        # neg = torch.sigmoid((center_vec * neg_vec).mean())

        # 3. 直接用cos训练
        pos = cosine_similarity(center_vec, pos_vec).sum() / self.sample_n / b / 2 + 0.5
        center_vec = self.w2v[center.repeat(1, 1, self.neg_rate).view(-1)]
        neg = cosine_similarity(center_vec, neg_vec).sum() / self.sample_n / self.neg_rate / b / 2 + 0.5
        # loss = sum(torch.log(expit(-1 * prod_term[1:])))  # for the sampled words
        print('pos:%.3f\tneg:%.3f' % (pos.item(), neg.item()))

        return 1 - pos + neg


if __name__ == '__main__':
    max_seq_len = 500
    sample_n = 400
    neg_rate = 10
    window = 1
    dataset = W2VDataset(vocab=vocab, sample_n=sample_n, max_seq_len=max_seq_len, window=window, neg_rate=neg_rate)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=40)

    model = SkipGramModel(vocab, hiddem_dim=100, sample_n=sample_n, neg_rate=neg_rate).train().cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3):
        for i, (center, pos_word, neg_word) in enumerate(tqdm(dataloader)):
            loss = model(center, pos_word, neg_word).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

        print('loss:', loss)
        torch.save(model.state_dict(), 'checkpoints/w2v/w2v.pth')
    # model.load_state_dict(torch.load(('checkpoints/w2v/w2v_min300.pth')))
    w2v = model.state_dict()['module.w2v']
    # print(w2v)
    # print(w2v.mean(), w2v.std())
    similarity = cosine_similarity(w2v[vocab['足球']].view(1, -1), w2v)
    most_similarity = torch.argsort(similarity, descending=True)[:10]
    least_similarity = torch.argsort(similarity, descending=False)[:10]

    for w in most_similarity:
        print(id2word[int(w)], similarity[w].item())
    print()
    for w in least_similarity:
        print(id2word[int(w)], similarity[w].item())

    # print(cosine_similarity(w2v[vocab['足球']].view(1, -1), w2v[vocab['篮球']].view(1, -1)))
    # print(cosine_similarity(w2v[vocab['足球']].view(1, -1), w2v[vocab['欧几里得']].view(1, -1)))
'''
思考：softmax可以抑制全部词，会更合理，cos拉扯力度太大，bias很大
'''
