import pandas as pd
import numpy as np

with open('dataset/data/train_set.csv') as f:
    train = f.readlines()[1:]
# test = pd.read_csv('data/test_a.csv')
with open('dataset/data/test_a.csv') as f:
    test = f.readlines()[1:]

from sklearn.model_selection import train_test_split

x, y = [], []
for line in train:
    y_, x_ = line.split('\t')
    x.append(x_.strip())
    y.append(y_.strip())

x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, random_state=0)

import os
import torch
import torch.nn.functional as F
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif
from sklearn import metrics
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

dataset = 'dataset'  # 数据集THUCNews
embedding = 'random'  # 'embedding_SougouNews.npz'  'random'
use_word = True  # 中文用char更好，不会有OOV+学习char分布比word分布更容易(因为量多）


def get_model(model_names):
    ret = []
    for model_name in model_names:
        x = import_module(name='models.' + model_name)  # 绝对导入
        config = x.Config(dataset, embedding)  # Config类
        config.train_path = dataset + '/data/dev.txt'
        vocab, train_data, dev_data, test_data = build_dataset(config, use_word)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)  # an object
        config.n_vocab = len(vocab)
        config.device = torch.device('cuda:0')
        model = x.Model(config).to(config.device)
        ret.append([model, config, dev_iter, test_iter])
    return ret


models = get_model(['TextRNN', 'TextCNN'])  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

models[0][0].load_state_dict(torch.load('dataset/saved_dict/TextRNN.ckpt_0.936358_59367.ckpt', map_location='cuda:0'))
models[1][0].load_state_dict(torch.load('dataset/saved_dict/TextCNN.ckpt_0.934816_23147.ckpt', map_location='cuda:0'))
# models[0][0].load_state_dict(torch.load('dataset/saved_dict/TextRNN.ckpt_0.937967_19153.ckpt', map_location='cuda:0'))
# models[1][0].load_state_dict(torch.load('dataset/saved_dict/TextCNN.ckpt_0.937892_74445.ckpt', map_location='cuda:0'))


def get_proba(models):
    ret = []
    # for model in models:
    m1, m2 = models
    model1, config, dev_iter, test_iter = m1
    model2, config, dev_iter, test_iter = m2

    model1.eval()
    model2.eval()

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model1(texts) + model2(texts)
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    f1 = f1_score(labels_all, predict_all, average='macro')
    print(f1)
    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    print("Precision, Recall and F1-Score...")
    print(report)
    ret.append([])
    return f1


def evaluate(models):
    pass


def generate_result(models, dev_f1):
    m1, m2 = models
    model1, config1, dev_iter, test_iter = m1
    model2, config2, dev_iter, test_iter = m2

    model1.eval()
    model2.eval()

    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model1(texts) + model2(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)

    with open('results/%s_result_%.4f.csv' % (config1.model_name + '+' +
                                              config2.model_name, dev_f1), 'w') as f:
        f.write('label\n')
        for p in predict_all:
            f.write(str(p) + '\n')


f1 = get_proba(models)
generate_result(models, f1)
# 0.9531669046695342)
'''
1024:0.95
2048:0.95
'''
