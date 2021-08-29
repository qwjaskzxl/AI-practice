from gensim.models import word2vec  # 必须按照[[],[]]才能保留原词！
import numpy as np
import pickle as pkl

w2v_size = 300
X = []
with open('dataset/data/corpus.txt', encoding='utf-8') as f:
    lines = f.readlines()[:]
    X = [[] for i in range(len(lines))]
    i = 0
    for line in lines:
        X[i] = line.split(' ')  # 由于这一整个是str，但是w2v要一个个词
        i += 1

k = None
model = word2vec.Word2Vec(X[:k], size=w2v_size, min_count=1, workers=-1, iter=10)
print(model)
# model.save('dataset/data/w2v/w2v_300_mincount1.model')
# model = word2vec.Word2Vec.load("dataset/data/w2v/w2v_300_mincount1.model")
# print(model)

V = pkl.load(open('dataset/data/vocab.pkl','rb'))
emb = np.random.randn(len(V), 300) * 0.000962582706903456
for w in V.keys():
    if w in model.wv.vocab:
        emb[V[w],:] = model.wv[w]
np.save('dataset/data/w2v_300.npy',emb)