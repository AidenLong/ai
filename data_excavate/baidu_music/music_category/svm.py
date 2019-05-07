# coding:utf-8
from sklearn import svm
from music_category import feature, acc
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

# from sklearn.grid_search import GridSearchCV

default_music_csv_file_path = '../data/music_features.csv'
default_model_file_path = '../data/music_model.pkl'
index_lable_dict = feature.fetch_index_label()

def poly(X, Y):
    """进行模型训练，并且计算训练集上预测值与label的准确性
    """
    clf = svm.SVC(kernel='poly', C=0.1, probability=True, decision_function_shape='ovo', random_state=0)
    clf.fit(X, Y)
    res = clf.predict(X)
    restrain = acc.get(res, Y)
    return clf, restrain


def fit_dump_model(train_percentage=0.7, fold=1, music_csv_file_path=None, model_out_f=None):
    """pass"""
    if not music_csv_file_path:
        music_csv_file_path = default_music_csv_file_path
    data = pd.read_csv(music_csv_file_path, sep=',', header=None, encoding='utf-8')

    max_train_source = None
    max_test_source = None
    max_source = None
    best_clf = None
    flag = True
    for index in range(1, int(fold) + 1):
        shuffle_data = shuffle(data)
        X = shuffle_data.T[:-1].T
        Y = np.array(shuffle_data.T[-1:])[0]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_percentage)
        (clf, train_source) = poly(x_train, y_train)
        y_predict = clf.predict(x_test)
        test_source = acc.get(y_predict, y_test)
        source = 0.35 * train_source + 0.65 * test_source
        if flag:
            max_source = source
            max_train_source = train_source
            max_test_source = test_source
            best_clf = clf
            flag = False
        else:
            if max_source < source:
                max_source = source
                max_train_source = train_source
                max_test_source = test_source
                best_clf = clf
        print('第%d次训练，训练集上的正确率为：%0.2f, 测试集上正确率为：%0.2f,加权平均正确率为：%0.2f' % (index, train_source, \
                                                                         test_source, source))
    print('最优模型效果：训练集上的正确率为：%0.2f,测试集上的正确率为：%0.2f, 加权评卷正确率为：%0.2f' % (max_train_source, \
                                                                      max_test_source, max_source))
    print('最优模型是：')
    print(best_clf)
    if not model_out_f:
        model_out_f = default_model_file_path
    joblib.dump(best_clf, model_out_f)


def load_model(model_f=None):
    if not model_f:
        model_f = default_model_file_path
    clf = joblib.load(model_f)
    return clf


def internal_cross_validation(X, Y):
    parameters = {
        'kernel': ('linear', 'rbf', 'poly'),
        'C': [0.1, 1],
        'probability': [True, False],
        'decision_function_shape': ['ovo', 'ovr']
    }
    clf = GridSearchCV(svm.SVC(random_state=0), param_grid=parameters, cv=5)  # 固定格式
    print('开始交叉验证获取最优参数构建')
    clf.fit(X, Y)
    print('最优参数：', end='')
    print(clf.best_params_)
    print('最优模型准确率：', end='')
    print(clf.best_score_)


def cross_validation(music_csv_file_path=None, data_percentage=0.7):
    if not music_csv_file_path:
        music_csv_file_path = default_music_csv_file_path
    print('开始读取数据：' + music_csv_file_path)
    data = pd.read_csv(music_csv_file_path, sep=',', header=None, encoding='utf-8')
    sample_fact = 0.7
    if isinstance(data_percentage, float) and 0 < data_percentage < 1:
        sample_fact = data_percentage
    data = data.sample(frac=sample_fact).T
    X = data[:-1].T
    Y = np.array(data[-1:])[0]
    internal_cross_validation(X, Y)


def fetch_predict_label(clf, X):
    label_index = clf.predict([X])
    label = index_lable_dict[label_index[0]]
    return label
