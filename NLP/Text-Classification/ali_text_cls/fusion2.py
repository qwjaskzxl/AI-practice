import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import f1_score

data_file = 'dataset/data/train_set.csv'


def softmax(x):
    e = np.exp(x)  # 除的话就是稀释了原始predict力度
    x = e / np.sum(e, axis=1, keepdims=True)
    return x


def fusion(models_name):
    # rnn = np.load(open('results/probs/TextRNN_result_0.9969_.npy', 'rb'))  # 0.9484
    rnn1 = np.load(open('results/probs/TextRNN_result_front.npy', 'rb'))  # 0.
    rnn2 = np.load(open('results/probs/TextRNN_result_back.npy', 'rb'))  # 0
    rnn = softmax(rnn1) + softmax(rnn2)

    # bert1 = np.load(open('results/probs/BERT_result_2.3100.npy', 'rb'))  # 0.9580
    bert1 = np.load(open('results/probs/BERT_result_2.2400_.npy', 'rb'))  # 0.
    bert2 = np.load(open('results/probs/BERT_result_2.2600_.npy', 'rb'))  # 0.
    bert = softmax(bert1) + softmax(bert2)

    # cnn_forward = np.load(open('results/probs/TextCNN_result_0.9958_.npy', 'rb'))  # 0.9412，
    # cnn_backward = np.load(open('results/probs/TextCNN_result_0.9907_.npy', 'rb'))  # TextCNN.ckpt_0.990660_91918.ckpt
    cnn1 = np.load(open('results/probs/TextCNN_result_0.9934.npy', 'rb'))  # 0.
    cnn2 = np.load(open('results/probs/TextCNN_result_back.npy', 'rb'))  # 0
    cnn = softmax(cnn1) + softmax(cnn2)

    # fasttext = np.load(open('results/probs/FastText_result_0.9343.npy', 'rb'))  # 这个可能分数太低了，毫无贡献，但是为啥也没拉低分数呢
    DL = rnn * 0.9 + cnn * 0.9 + bert

    print(rnn)
    print(cnn)
    print(bert)

    tfidf_sgdc = np.load(open('results/TFIDF+SGDC_prob_hard.npy', 'rb')) + np.load(open('results/TFIDF+SGDC_back_prob_hard.npy', 'rb'))
    tfidf_lsvc = np.load(open('results/TFIDF+LSVC_prob.npy', 'rb')) + np.load(open('results/TFIDF+LSVC_back_prob.npy', 'rb'))
    ML = tfidf_sgdc * 0.5 + tfidf_lsvc * 0.5
    # print(tfidf_lsvc)
    print(ML)

    models_sc = DL + ML
    predict_all = np.argmax(models_sc, axis=1)
    print(predict_all)

    # models_name = 'TextCNN+TextRNN+FastText'
    with open('results/%s_result.csv' % (models_name), 'w') as f:
        f.write('label\n')
        for p in predict_all:
            f.write(str(p) + '\n')


class Voting(nn.Module):
    def __init__(self):
        super(Voting, self).__init__()
        # self.W = nn.Linear(3, 1, bias=False)
        self.W = nn.Parameter(torch.ones(2, 1))

    def forward(self, x):
        return torch.matmul(x, self.W).squeeze(2)


def get_W():
    b = 256
    N = 200000
    n = N // b + 1
    y = np.zeros(N).astype(int)
    with open('dataset/data/train_all.txt', 'r') as f:
        for i, ins in enumerate(f.readlines()):
            y[i] = ins.split('\t')[1]

    # y = np.array(pd.read_csv(data_file, sep='\t', encoding='UTF-8')['label'])
    y_ = torch.LongTensor(y).cuda()

    rnn = np.load(open('results/probs/TextRNN_result_0.9916.npy', 'rb')).astype(np.float32)[:, :, np.newaxis]  # 0.9484
    cnn1 = np.load(open('results/probs/TextCNN_front_0.9901.npy', 'rb')).astype(np.float32)[:, :, np.newaxis]  # 0.9412，
    cnn2 = np.load(open('results/probs/TextCNN_back_0.9803.npy', 'rb')).astype(np.float32)[:, :, np.newaxis]  # TextCNN.ckpt_0.990660_91918.ckpt
    x = torch.from_numpy(np.concatenate([rnn, cnn1], axis=2)).cuda()
    # x = rnn + cnn1 + cnn2 #这个反向的CNN拖累了模型

    model = Voting()
    for p in model.parameters():
        print(p)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    CELoss = nn.CrossEntropyLoss()

    model.train().cuda()
    predict_all = np.array([], dtype=int)
    for i in range(n):
        bound = min(i * b + b, N)
        optim.zero_grad()
        out = model(x[i * b:bound])
        loss = CELoss(out, y_[i * b:bound])
        loss.backward()
        optim.step()

        predict = torch.max(out.data, 1)[1].cpu().numpy()
        predict_all = np.append(predict_all, predict)
        # print(f1_score(predict, y[i * b:bound], average='macro'))

    print(f1_score(predict_all, y, average='macro'))
    for p in model.parameters():
        print(p)


if __name__ == '__main__':
    # e = np.array([[1, 2, 3], [3, 4, 5]])
    # print(e.shape)
    # print(np.exp(e))
    # print(np.sum(np.exp(e), axis=1, keepdims=True))
    # print(np.exp(e) / np.sum(np.exp(e), axis=1, keepdims=True))

    fusion('TextCNN双向+TextRNN+BERT+SGDC')
    # TODO：加个静态的词向量，放rnn cnn里面训练两个方向OK的
    # get_W()
