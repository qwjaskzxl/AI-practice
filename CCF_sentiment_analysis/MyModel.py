# coding=UTF-8
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
import lightgbm as lgb

from sklearn.externals import joblib  # to save


class Model():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    def lr(self, to_fit=False):
        clf = LogisticRegression(C=4, dual=True)
        if to_fit == True:
            clf.fit(self.X_train, self.y_train)
            joblib.dump(clf, './models/lr.m')
        return clf

    def svc(self, to_fit=False, get_param=False):
        clf = svm.SVC(C=1.2, kernel='linear', tol=0.01, max_iter=800)  # tol：停止训练的误差精度
        if get_param == True:
            param_grid = {
                'C': np.linspace(0.1, 5, 10000),
                'tol': np.linspace(0.1,5, 10000),
            }
            return param_grid
        if to_fit == True:
            clf.fit(self.X_train, self.y_train)
            joblib.dump(clf, './models/svc.m')
        return clf

    def lsvc(self, to_fit=False, get_param=False):
        clf = svm.LinearSVC(C=6, loss='squared_hinge', tol=8, max_iter=1000)
        if get_param == True:
            param_grid = {
                'C': np.linspace(2, 8, 10000),
                'tol': np.linspace(2, 8, 10000),
            }
            return param_grid
        if to_fit == True:
            clf.fit(self.X_train, self.y_train)
            joblib.dump(clf, './models/lsvc.m')
        # {'tol': 3.367, 'C': 6.534} 0.680 {2.75,9.79} 0.686' tol': 4.78, 'C': 0.616 0.694
        # {'tol': 5.80 'C': 0.95}.708(std:0.017)
        return clf

    def rf(self, to_fit=False):
        clf = RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_split=2, min_samples_leaf=1,
                                     n_jobs=-1)
        if to_fit == True:
            clf.fit(self.X_train, self.y_train)
            joblib.dump(clf, './models/rf.m')
        return clf

    def sgdc(self, to_fit=False, get_param=False):
        clf = SGDClassifier(alpha=0.0003, loss="squared_hinge", epsilon=0.1,
                            max_iter=1000, tol=0.4, penalty='l2',
                            n_jobs=-1)  # penalty=”elasticnet”: L2和L1的convex组合; (1 - l1_ratio) * L2 + l1_ratio * L1
        if get_param == True:
            params = {
                'alpha': np.linspace(0.0001, 0.0006, 10000),
                # 'epsilon': np.linspace(0.05, 0.5, 200),
                'tol': np.linspace(0.1, 10, 10000),
                # 'epsilon':[0.1],
                # 'tol':np.linspace(0.001,0.1,20),
            }
            return params
        if to_fit == True:
            clf.fit(self.X_train, self.y_train)
            joblib.dump(clf, './models/sgdc.m')
        return clf

    def lgb(self, to_fit=''):
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_val = lgb.Dataset(self.X_val, self.y_val)
        params = {
            'learning_rate': 0.1,  # 默认0.1
            'boosting_type': 'gbdt',  # 默认gbdt
            'max_depth': 100,
            'num_leaves': 80,  # 默认31
            'min_data_in_leaf': 20,  # 默认20
            'objective': 'multiclass',
            'num_class': 3,
            'device': 'cpu',
        }
        clf = lgb.train(params, lgb_train, num_boost_round=40, valid_sets=lgb_val)
        return clf

class Lgb_model(Model):  # 自动继承model的构造函数
    def __init__(self, X, y):
        super(Lgb_model, self).__init__(X, y)
