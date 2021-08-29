# coding=utf-8
from tqdm import tqdm
import json
import logging
import multiprocessing
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def build_corpus(fname):
    from gensim.corpora import WikiCorpus
    import zhconv, jieba, re

    wiki = WikiCorpus(fname, lemmatize=False, dictionary={})
    with open('data/wiki_corpus_tmp.txt', 'w') as f:
        for text in tqdm(wiki.get_texts()):
            text = re.sub(r'[^\u4e00-\u9fa5]', '', ''.join(text))
            text = zhconv.convert(text, 'zh-hans')
            text = ' '.join(jieba.cut(text))
            f.write(text + '\n')
            # print(text)
            # exit()


def build_w2v(fname):
    from gensim.models.word2vec import LineSentence
    from gensim.models import Word2Vec
    # X = LineSentence(fname)
    k = 50000
    with open(fname, 'r', encoding='utf-8') as f:
        corpus = f.readlines()[:k]
        X = [[] for _ in range(len(corpus))]
        for i, s in enumerate(corpus):
            X[i] = s.strip().split(' ')
    model = Word2Vec(X, size=100, min_count=10, workers=multiprocessing.cpu_count(), iter=3)
    model.save('data/word2vec/w2v.model')
    # model = Word2Vec.load(fname)
    with open('data/wiki_vocab_min10.json', 'w') as f:
        words = model.wv.vocab.keys()
        print('词数:', len(words))
        vocab = {}
        for w in words:
            vocab[w] = len(vocab)
        json.dump(vocab, f)
        # print(vocab)

    # print(model.wv.get_vector("数学"))
    print(model.wv.most_similar(positive=['足球'], topn=10))


if __name__ == '__main__':
    # build_corpus('data/zhwiki-latest-pages-articles.xml.bz2')
    build_w2v('data/wiki_corpus.txt')