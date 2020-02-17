# coding=UTF-8
from MyModel import Model
from time import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Evaluator(Model):
    def __init__(self, X, y):
        super(Evaluator, self).__init__(X, y)  # super会查找所有的超类们，以及超类们的超类们

    # def __getitem__(self, item):
    #     pass

    def cross_val(self, clf):
        Kfold = KFold(n_splits=5, shuffle=True, random_state=None)  # shuffle作用：只是打乱，这个貌似没有stratify功能啊？？
        scores = cross_val_score(clf, self.X, self.y, cv=Kfold, scoring='f1_macro', n_jobs=-1)
        return scores

    def para_search_report(self, results, n_top=8):  # 忘写self会“object is not subscriptable”的bug，就还要写getitem构造函数
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)  # 保留正的？
            # candidates = np.array(results['rank_test_score'] == i)
            for candidate in candidates:
                print(i)
                print("Mean validation score:%.3f(std:%.3f)" %
                      (results['mean_test_score'][candidate],
                       results['std_test_score'][candidate]))
                print("Parameters:{0}".format(results['params'][candidate]))
                # print(i)

    def param_search(self, clf, param, rand_search=False):
        a = time()
        Kfold = KFold(n_splits=5, shuffle=True, random_state=None)
        if rand_search == True:
            search = RandomizedSearchCV(clf, param_distributions=param, cv=Kfold,
                                        n_iter=100, scoring='f1_macro', n_jobs=-1)
        else:
            search = GridSearchCV(clf, param_grid=param, cv=Kfold, scoring='f1_macro', n_jobs=-1)
        search.fit(self.X, self.y)
        print('搜索参数耗时：%.1fs' % (time() - a))
        return search

    def clf_report(self, clf, X_test=None, y_test=None):
        if X_test != None:
            pred = clf.predict(self.X)
            rep_tra = metrics.classification_report(self.y, pred)
            pred = clf.predict(X_test)
            rep_test = metrics.classification_report(y_test, pred)
            return (rep_tra, rep_test)

        elif X_test == None:
            pred = clf.predict(self.X_train)
            rep_tra = metrics.classification_report(self.y_train, pred)
            pred = clf.predict(self.X_val)
            rep_val = metrics.classification_report(self.y_val, pred)
            return (rep_tra, rep_val)

    def lgb_clf_report(self, clf):
        pred_prob = clf.predict(self.X_train)
        pred = np.argmax(pred_prob, axis=1)  # 获最大概率对应的label
        rep_tra = metrics.classification_report(self.y_train, pred)
        pred_prob = clf.predict(self.X_val)
        pred = np.argmax(pred_prob, axis=1)  # 获最大概率对应的label
        rep_val = metrics.classification_report(self.y_val, pred)
        return (rep_tra, rep_val)
