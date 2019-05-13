# -- encoding:utf-8 --
from surprise import Dataset
from surprise import SVD
from surprise import GridSearch

"""
NOTE: 如果安装的是numpy+mkl的库的话，必须将numpy的core文件夹中的所有以mkl_开头的文件(动态链接库dll文件)全部放到Python的根目录下(即python.exe所在的文件夹中)
"""
if __name__ == '__main__':
    # 1. 加载数据
    data = Dataset.load_builtin(name='ml-100k')

    # 2. 对数据做一个K-Fold
    data.split(n_folds=3)

    # 3. 定义模型参数选择列表
    param_grid = {
        'n_epochs': [10, 20],  # 给定迭代次数
        'lr_all': [0.01, 0.05],  # 给定全局默认学习率
        'reg_all': [0.01, 0.05],  # 给定全局的正则化项系数
        'lr_bu': [0.001, 0.005]  # 单独给定bu迭代计算过程中的学习率
    }

    # 4. 定义网格搜索对象
    """
    __init__(self, algo_class, param_grid, measures=['rmse', 'mae'],
                     n_jobs=-1, pre_dispatch='2*n_jobs', seed=None, verbose=1,
                     joblib_verbose=0)
    algo_class: 这里指定的是推荐算法模型的API/Class名称，不能是算法模型对象 
    param_grid: 网格选择过程中的参数列表; 必须是一个key/value的字典对象，key必须是algo_class算法模型对应的参数名称，value必须是一个list的集合，集合中的元素是模型参数对应的取值
    measures：指定最优模型的评估方式                
    """
    grid_search = GridSearch(algo_class=SVD, param_grid=param_grid, measures=['rmse', 'mae', 'FCP'])

    # 5. 模型训练，找出最优参数
    grid_search.evaluate(data)

    # 6. 输出最优的模型参数以及最优情况下的最优指标值
    # 6.1 输出最优的RMSE模型
    print("*" * 20)
    print(grid_search.best_score['RMSE'])
    print(grid_search.best_params['RMSE'])
    # 6.2 输出最优的MAE模型
    print("*" * 20)
    print(grid_search.best_score['MAE'])
    print(grid_search.best_params['MAE'])
    # 6.1 输出最优的FCP模型
    print("*" * 20)
    print(grid_search.best_score['FCP'])
    print(grid_search.best_params['FCP'])
