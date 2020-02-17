import time
import cython
import numpy as np
from gensim.models import word2vec  # 必须按照[[],[]]才能保留原词！
from gensim.models.word2vec import LineSentence
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
w2v_size = 10
corpus_path = './dataset/corpus.txt'
model_path = './models/w2v/w2v_100_min3_nostopword.model'
# model_path = './models/w2v_100_min3.model'

def train_w2v():
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:]
        global X
        X = [[] for i in range(len(lines))]
        for i, line in enumerate(lines):
            X[i] = line.split(' ')

    a = time.time()
    model = word2vec.Word2Vec(X, size=w2v_size, min_count=3, workers=40)
    model.save(model_path)
    print(model)
    print("Time consumption on training:", time.time() - a)

def play():
    model = word2vec.Word2Vec.load(model_path)
    for e in model.most_similar(positive=['中国'], topn=10):
        print(e[0], e[1])


def show():
    model = word2vec.Word2Vec.load(model_path)
    print(model)
    words = list(model.wv.vocab)[:150]
    X = model[model.wv.vocab][:150, :]  # 词向量矩阵
    print(X.shape)
    # 降维
    pca = PCA(n_components=2)
    X_2 = pca.fit_transform(X)
    print(X_2.shape)

    plt.scatter(X_2[:, 0], X_2[:, 1])  # 散点图的xy坐标
    for i, word in enumerate(words):
        plt.annotate(word, xy=(X_2[i, 0], X_2[i, 1]))
    plt.show()


# train_w2v()
# show()
play()