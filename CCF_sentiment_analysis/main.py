# coding=UTF-8
from PreProsess import Dataset, Feature
from MyModel import Model
from Evaluate import Evaluator
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = './dataset'
corpus_path = '/new_corpus_havestp_C.txt'
trainset_path = '/new_trainset_havestp_C.txt'
testset_path = '/new_testset_havestp_C.txt'


def create_Xy():
    # 保存语料库、分词后的数据集
    trainset0, label = data.get_trainset0()  # 自动解包
    trainset = data.cutword_list(trainset0)
    data.save_txt(trainset, '/new_trainset_havestp_C.txt')
    """
    X_train,X_val,y_train,y_val = train_test_split(trainset, list(label), test_size=0.1, shuffle=True, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.15, shuffle=True, random_state=0)
    for i in range(len(X_train)):
        X_train[i] += '\t'+str(y_train[i])
    data.save_txt(X_train, '/train.txt')
    for i in range(len(X_test)):
        X_test[i] += '\t'+str(y_test[i])
    for i in range(len(X_val)):
        X_val[i] += '\t'+str(y_val[i])
    data.save_txt(X_val, '/dev.txt')
    data.save_txt(X_test, '/test.txt')
    """

    test_label, testset0 = data.get_testset0()# y_id, testset0 = data.get_testset0()
    testset = data.cutword_list(testset0)
    data.save_txt(testset, '/new_testset_havestp_C.txt')
    corpus = trainset + testset
    data.save_txt(corpus, '/new_corpus_havestp_C.txt')
    print(len(corpus), len(trainset), len(testset))
    # 14621 7265 7356

def output(y_id, pred):
    with open('./solution/ans.csv', 'w', encoding='utf-8') as f:
        f.write('id,label' + '\n')  # 自动换行
        for i in range(len(y_id)):
            f.write(str(y_id[i]) + ',' + str(pred[i]) + '\n')


if __name__ == '__main__':
    # 读取数据
    data = Dataset(DATA_PATH)
    # create_Xy()
    corpus = data.read_txt(corpus_path)[:]
    trainset = data.read_txt(trainset_path)
    testset = data.read_txt(testset_path)
    print(len(corpus), len(trainset), len(testset))

    # 特征工程
    feat = Feature(trainset)  # 用全部语料fit的tfidf
    vect_fit = feat.tfidf_fit()
    X = feat.tfidf_trans(vect_fit, trainset)
    print('X\'s tfidf shape:', X.shape)
    y = data.get_trainset0()[1]
    model = Model(X, y)
    eva = Evaluator(X, y)

    # 训练+评估
    clf, param = model.lsvc(to_fit=True), model.lsvc(get_param=True)
    print(clf)
    # search = eva.param_search(clf, param, rand_search=True)
    # print(search.best_params_, search.best_score_)
    # print(eva.para_search_report(pd.DataFrame(search.cv_results_)))
    print(eva.cross_val(clf).mean())
    print(*eva.clf_report(clf, X_test=feat.tfidf_trans(vect_fit, testset),
                          y_test=data.get_testset0()[0]))

    # 提交
    # clf = model.lsvc(to_fit=True)
    # clf.fit(X, y)
    # X_test = feat.tfidf_trans(vect_fit, testset)
    # y_id = data.get_testset0()[0]
    # output(y_id, clf.predict(X_test))
