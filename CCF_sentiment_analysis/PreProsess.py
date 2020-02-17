# coding=UTF-8
import pandas as pd
from time import time
import jieba
import os
from sklearn.feature_extraction.text import TfidfVectorizer


class Dataset():  # 括号里写父类
    """
    get_trainset0:获取原始训练集(匹配id)
    get_testset0:获取原始测试集
    cutword_list:分词(类内调用)
    save_txt:保存数据到txt文件
    0表示没分词的句子
    """

    def __init__(self, path):
        self.path = path

        df_train = pd.read_csv(path + '/new_trainset.csv').dropna(axis=0, how='any')  # 在0这个维度上（也就是列，即删除一行
        df_train["content"] = df_train[["title", "content"]].apply(lambda x: "".join([str(x[0]), str(x[1])]), axis=1)
        # label = pd.read_csv(path + '/Train_DataSet_Label.csv')
        # self.df_train = pd.merge(df_train, label, how='inner', on='id')
        self.df_train = df_train

        df_test = pd.read_csv(path + '/new_testset.csv')
        df_test["content"] = df_test[["title", "content"]].apply(lambda x: "".join([str(x[0]), str(x[1])]), axis=1)
        self.df_test = df_test
        self.stopwords = {}
        with open(path + '/stopwords', encoding='utf-8') as f:
            for line in f.readlines():
                self.stopwords[line.strip()] = 1  # 直接读取末尾有换行符，要去除

    def get_trainset0(self):
        # trainset0 = []
        # label = []
        # label_id = {}
        # for i in range(len(self.label)): #list:6.7s，dict:0.4s!
        #     label_id[self.label['id'][i]] = 1
        # for x in self.df_train.iterrows():
        #     if(x[1]['id'] in label_id):
        #         trainset0.append(str(x[1]['title'])+';'+str(x[1]['content']))
        #         label.append(self.label['label'][i])
        # trainset0 = pd.merge(self.df_train, self.label, how='inner', on='id')
        # print(trainset0.columns)
        return ((self.df_train['content']), self.df_train['label'])

    def get_testset0(self):
        return (self.df_test['label'], self.df_test['content']) #原本是id和content

    def cutword_list(self, dataset0):
        # 传入必须是list/Series等容器
        a = time()
        cutword = []
        for sentence in dataset0[0:]:
            sentence = sentence.replace('\n', ' ').replace('\r', ' ')
            words, sente_Chinese = '', ''
            for uchar in sentence:
                if uchar >= u'\u4e00' and uchar <= u'\u9fa5':#只保留汉字
                    sente_Chinese += uchar
            for word in jieba.cut(sente_Chinese):
                # if word not in self.stopwords:#list:197.4s,dict:51s，不去停用词：50s
                words += word + ' '
            cutword.append(words)
        print('分词耗时：%.1fs' % (time() - a))
        return cutword

    def save_txt(self, data, file):
        with open(self.path + file, 'w', encoding='utf-8') as f:
            for sente in data:
                f.write(sente.replace('\n', '') + '\n')

    def read_txt(self, file):
        if os.path.exists(self.path + file):
            with open(self.path + file, 'r', encoding='utf-8') as f:
                return f.readlines()
        else:
            print('%s不存在'%(self.path + file))

class Feature():
    def __init__(self, data_fit):
        self.data_fit = data_fit

    def tfidf_fit(self):
        a = time()
        vect = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.95, use_idf=1, smooth_idf=1, sublinear_tf=1)
        vect.fit(self.data_fit)
        print('tfidf_fit耗时：%.1fs' % (time() - a))
        return vect

    def tfidf_trans(self, vect, data_trans):
        a = time()
        tfidf = vect.transform(data_trans)
        print('tfidf_trans耗时：%.1fs' % (time() - a))
        return tfidf