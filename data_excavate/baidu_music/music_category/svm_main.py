# coding:utf-8
from baidu_music.music_category import svm
from baidu_music.music_category import feature

svm.cross_validation(data_percentage=0.99)

svm.fit_dump_model(train_percentage=0.9, fold=100)
path = '../data/test/你的选择.mp3'
music_feature = feature.extract(path)
clf = svm.load_model()
label = svm.fetch_predict_label(clf, music_feature)
print('预测标签为：%s' % label)

# path = ['../data/test/Lasse Lindh - Run To You.mp3', '../data/test/Maize - I Like You-浪漫.mp3',
#         '../data/test/孙燕姿 - 我也很想他 - 怀旧.mp3']
# for index in path:
#     #     print(str(index))
#     music_feature = feature.extract(index)
#     clf = svm.load_model()
#     label = svm.fetch_predict_label(clf, music_feature)
#     print('预测标签为：%s' % label)
