# -- encoding:utf-8 --
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, StringIndexer
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Word2Vec
from pyspark.ml.feature import PolynomialExpansion, OneHotEncoder
from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import RFormula, PCA

# 给定SPARK_HOME环境变量
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'


def tokenizer(sentence_df, split_column_name, word_column_name):
    """
    对给定的DataFrame中对应列的数据采用Tokenizer方式进行文本数据的划分
    :param sentence_df:  输入的DataFrame
    :param split_column_name:  输入的划分数据列名称
    :param word_column_name:  给定进行文本数据划分后的DataFrame中的单词对应的列名称，要求列不存在
    :return:
    """
    # 1. 构建模型
    """
    inputCol=None, 给定输入的列名称
    outputCol=None 给定输出的列名称
    """
    sentence_tokenizer = Tokenizer(inputCol=split_column_name, outputCol=word_column_name)
    # 2. 对数据做转换
    result_df = sentence_tokenizer.transform(sentence_df)
    # 3. 结果返回
    return result_df


def regex_tokenizer(sentence_df, split_column_name, word_column_name):
    """
    对给定的DataFrame中对应列的数据采用RegexTokenizer方式进行文本数据的划分
    :param sentence_df:  输入的DataFrame
    :param split_column_name:  输入的划分数据列名称
    :param word_column_name:  给定进行文本数据划分后的DataFrame中的单词对应的列名称，要求列不存在
    :return:
    """
    # 1. 构建模型
    """
    minTokenLength=1, 最终划分出来的单词，最小长度，如果小于该长度的单词，直接被删除掉
    gaps=True, 标记符，功能：给定pattern正则字符串的作用，当gaps为True的时候，pattern给定的是分割符对应的正则字符串；如果为False的时候，pattern给定的是数据对应的正则字符串
    pattern="\\s+", 给定正则字符串
    inputCol=None, 给定输入的列名称
    outputCol=None 给定输出的列名称
    """
    sentence_regex_tokenizer = RegexTokenizer(minTokenLength=2, gaps=False, pattern='\\w+', \
                                              inputCol=split_column_name, outputCol=word_column_name)
    # 2. 对数据做转换
    result_df = sentence_regex_tokenizer.transform(sentence_df)
    # 3. 结果返回
    return result_df


def remove_stopwords(word_df, word_column_name, filter_word_column_name):
    """
    对给定的DataFrame中的对应列数据单词做停止词过滤的操作
    :param sentence_df:  输入的DataFrame
    :param word_column_name:  给定的输入的列名称
    :param filter_word_column_name: 给定过滤之后对应的列名称
    :return:
    """
    # 1. 构建模型
    """
    stopWords=None, 给定停止词列表， 默认为：http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
    caseSensitive=False 进行单词过滤的时候是否考虑大小写
    inputCol=None, 给定输入的列名称
    outputCol=None 给定输出的列名称
    """
    stop_words_remover = StopWordsRemover(inputCol=word_column_name, outputCol=filter_word_column_name,
                                          stopWords=['and', 'hi', 'i', 'about', 'are'])
    # 2. 对数据做转换
    result_df = stop_words_remover.transform(word_df)
    # 3. 结果返回
    return result_df


def string_indexer(category_df, input_string_category_column_name, output_indexer_column_name):
    """
    对于类别数据给定数字编号，比那好按照出现的次数从多到少给定序号
    :param category_df:  输入的DataFrame
    :param input_string_category_column_name:  输入的字符串类型的列名称
    :param output_indexer_column_name:  给定最终输出的索引列名称
    :return:
    """
    # 1. 构建模型
    """
    inputCol=None, 给定输入的列名称
    outputCol=None 给定输出的列名称
    handleInvalid="error" 设置当进行category类别转换的时候，如果模型中不存在对应的类别，那么做何种操作？可选参数：error(抛出异常)、skip(数据过滤)
    """
    string_indexer = StringIndexer(inputCol=input_string_category_column_name, outputCol=output_indexer_column_name,
                                   handleInvalid='skip')
    # 2. 模型训练
    string_indexer_model = string_indexer.fit(category_df)
    # 3. 对数据做转换
    result_df = string_indexer_model.transform(category_df)
    # 3. 结果返回
    return string_indexer_model, result_df


def count_tf(word_tf):
    """
    对文本数据进行count tf的转换，即词袋法
    :param word_tf: 给定的DataFrame， 要求输入的列类型必须是单词组成的vector向量类型
    :return:
    """
    # 1. 算法构建
    """
    minTF=1.0 => 当单词的TF值大于等于该值的时候，该单词才会统计其出现的次数
    minDF=1.0 => 当单词的DF值大于等于该值的时候，该单词才会统计其出现的次数
    vocabSize=1 << 18 => 允许最多的特征属性是多少
    inputCol=None => 输入列名称，要求列数据类型是Vector
    outputCol=None => Count TF转换之后的输出DataFrame中对应的列名称，要求列不存在
    """
    counter = CountVectorizer(inputCol='word', outputCol='features')
    # 2. 模型训练(计算出具有那些单词作为特征属性)
    counter_model = counter.fit(word_tf)
    # 3. 使用模型对数据做转换操作
    result_df = counter_model.transform(word_tf)
    # 4. 结果返回
    return result_df


def hash_tf(word_tf):
    """
    对文本数据进行hash tf的转换
    :param word_tf: 给定的DataFrame， 要求输入的列类型必须是单词组成的vector向量类型
    :return:
    """
    # 1. 算法构建
    """
    numFeatures=1 << 18 => 给定我们转换之后的特征属性的数量
    inputCol=None => 输入列名称，要求列数据类型是Vector
    outputCol=None => Count TF转换之后的输出DataFrame中对应的列名称，要求列不存在
    """
    hash = HashingTF(numFeatures=10, inputCol='word', outputCol='features')
    # 2. 使用模型对数据做转换操作
    result_df = hash.transform(word_tf)
    # 3. 结果返回
    return result_df


def tf_idf(word_tf_df):
    """
    对做完TF转换的DataFrame数据做一个IDF转换
    :param word_tf: 给定的DataFrame， 要求输入的列类型必须是单词的TF值组成的vector向量类型
    :return:
    """
    # 1. 算法构建
    """
    minDocFreq=0, 当特征值大于等于该值的时候，才会计算IDF，否则直接赋值为0
    inputCol=None => 输入列名称，要求列数据类型是Vector
    outputCol=None => TF-IDF转换之后的输出DataFrame中对应的列名称，要求列不存在
    """
    idf = IDF(minDocFreq=1, inputCol='features', outputCol='features2')
    # 2. 模型训练(计算出具有那些单词作为特征属性)
    idf_model = idf.fit(word_tf_df)
    # 3. 使用模型对数据做转换操作
    result_df = idf_model.transform(word_tf_df)
    # 4. 结果返回
    return result_df


def word_2_vec(word_tf):
    """
    对文本数据进行Word2Vec转换
    :param word_tf: 给定的DataFrame， 要求输入的列类型必须是单词组成的vector向量类型
    :return:
    """
    # 1. 算法构建
    """
    vectorSize=100, 转换成为的词向量中的维度数目，默认为100
    minCount=5, 在转换过程中，单词至少出现的数量，默认为5
    maxIter=1,模型构建过程中的迭代次数，默认为1
    inputCol=None => 输入列名称，要求列数据类型是Vector
    outputCol=None => Count TF转换之后的输出DataFrame中对应的列名称，要求列不存在
    """
    w2v = Word2Vec(vectorSize=5, minCount=1, maxIter=10, inputCol='word', outputCol='features')
    # 2. 模型训练(计算出具有那些单词作为特征属性)
    w2v_model = w2v.fit(word_tf)
    # 3. 使用模型对数据做转换操作
    result_df = w2v_model.transform(word_tf)
    # 4. 结果返回
    return result_df


def poly_expan(df):
    # 1. 算法对象构建
    poly = PolynomialExpansion(degree=2, inputCol='value', outputCol='features')
    # 2. 数据转换
    result_df = poly.transform(df)
    return result_df


def one_hot_encoder(category_index_df):
    # 1. 算法对象构建
    # dropLast: 指定是否删除最后一列，默认是删除
    one_hot = OneHotEncoder(dropLast=False, inputCol='category_indexer', outputCol='features')
    # 2. 数据转换
    result_df = one_hot.transform(category_index_df)
    return result_df


def normalizer(df):
    """
    对数据做一个归一化操作
    :param df:
    :return:
    """
    # 1. 算法对象构建
    # p: 进行归一化的时候，分母进行什么样子的操作，设置为2.0表示使用当前样本对应的特征属性对应的二范式作为分母(所有特征值的平方和开根号)
    norm = Normalizer(p=2.0, inputCol='value', outputCol='features')
    # 2. 数据转换
    result_df = norm.transform(df)
    return result_df


def standard_scaler(df):
    # 1. 构建算法对象
    # withMean=False, withStd=True, inputCol=None, outputCol=None
    ss = StandardScaler(withMean=True, withStd=True, inputCol='value', outputCol='features')
    # 2. 需要计算均值和标准差(模型训练)
    ss_model = ss.fit(df)
    # 3. 数据转换
    result_df = ss_model.transform(df)
    return result_df


def max_min_scaler(df):
    # 1. 构建算法对象
    # min=0.0, max=1.0, inputCol=None, outputCol=None
    mms = MinMaxScaler(min=0.0, max=5.0, inputCol='value', outputCol='features')
    # 2. 需要计算最大值和最小值(模型训练)
    mms_model = mms.fit(df)
    # 3. 数据转换
    result_df = mms_model.transform(df)
    return result_df


if __name__ == '__main__':
    # 1. 创建上下文
    conf = SparkConf() \
        .setMaster('local[10]') \
        .setAppName('feature_process')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sparkContext=sc)

    # 2. 构建一个RDD并将其转换为DataFrame
    sentence_rdd = sc.parallelize([
        "Hi I heard about spark",
        "I wish java And could use case classes",
        "Logistic regression models are neat",
        "Softmax regression models and lasso regression"
    ]).map(lambda t: Row(sentence=t))
    category_rdd = sc.parallelize([
        "a", "b", "c", "d", "a", "b", "b", "c", "d", "b"
    ]).map(lambda t: Row(category=t))
    category_rdd_test = sc.parallelize([
        "a", "b", "c", "d", "e"
    ]).map(lambda t: Row(category=t))
    sentence_df = sqlContext.createDataFrame(sentence_rdd)
    category_df = sqlContext.createDataFrame(category_rdd)
    category_df_test = sqlContext.createDataFrame(category_rdd_test)
    sentence_df.show(truncate=False)

    # ---------------开始数据处理------------------------------
    # 一、文本数据分词
    """
    可以使用jieba来分词；也可以使用SparkCore、SparkSQL根据数据的特征对数据做分词；还可以使用spark mllib自带的API来作词
    """
    # 1.1 Tokenizer: 默认按照空格对DataFrame中的对应列的数据进行数据的划分
    tokenizer_result_df = tokenizer(sentence_df, 'sentence', 'words')
    print(tokenizer_result_df.schema)
    tokenizer_result_df.show(truncate=False)
    # 1.2 RegexTokenizer: 根据给定的正则的相关字符串参数，对对应列数据做文本单词的划分
    regex_tokenizer_result_df = regex_tokenizer(sentence_df, 'sentence', 'words')
    regex_tokenizer_result_df.show(truncate=False)
    # 1.3 StopWordsRemover：删除给定DataFrame中的停止词
    stop_words_remove_result_df = remove_stopwords(tokenizer_result_df, 'words', 'filter_words')
    stop_words_remove_result_df.show(truncate=False)
    # 1.4 String类型的类别转换为数值类型的数据
    string_indexer_model, category_result_df = string_indexer(category_df, 'category', 'category_indexer')
    category_result_df.show(truncate=False)
    string_indexer_model.transform(category_df_test).show(truncate=False)

    # 二、单词转换为特征向量
    """
    解决方案：
    1. 词袋法：以单词或者单词的hash值作为特征属性，然后计算当前文本再各个特征属性上出现的数量作为特征属性值，从而统计出当前文本对应的特征向量
    2. TF-IDF：在词袋法的基础上，加入单词的逆文档频率
    3. Word2Vec：将单词数据做一个哑编码，然后将哑编码之后的向量作为样本特征输入到神经网络中，最终输出一个给定维度大小的特征向量
    """
    word_df = stop_words_remove_result_df.select('filter_words').toDF('word')
    word_df.show(truncate=False)
    # 2.1 count tf操作
    word_count_result_df = count_tf(word_df)
    word_count_result_df.show(truncate=False)
    # 2.2 hash tf操作
    word_hash_result_df = hash_tf(word_df)
    word_hash_result_df.show(truncate=False)
    # 2.3 TF-IDF操作
    tf_idf_result_df = tf_idf(word_count_result_df)
    tf_idf_result_df.show(truncate=False)
    # 2.4 Word2Vec操作
    word_2_vec_result_df = word_2_vec(word_df)
    word_2_vec_result_df.show(truncate=False)

    # 三、特征转换
    """
    主要操作：
    1. 直接使用SparkCore/SparkSQL的相关API对数据进行转换操作，比如: 指数化、对数化、区间化/分区/分桶.....
    2. 使用Spark MLlib提供的专门进行数据特征转换的API进行操作，比如：多项式扩展、哑编码、归一化、标准化、区间缩放法等
    """
    double_vector_value_df = sqlContext.createDataFrame(sc.parallelize([
        Vectors.dense([1.0, 2.0, 3.0]),
        Vectors.dense([4.0, 5.0, 6.0]),
        Vectors.dense([7.0, 8.0, 9.0])
    ]).map(lambda t: Row(value=t)))
    # 3.1 多项式扩展
    poly_result_df = poly_expan(double_vector_value_df)
    poly_result_df.show(truncate=False)
    # 3.2 哑编码
    one_hot_result_df = one_hot_encoder(category_result_df)
    one_hot_result_df.show(truncate=False)
    # 3.3 归一化处理
    norm_result_df = normalizer(double_vector_value_df)
    norm_result_df.show(truncate=False)
    # 3.4 标准化处理
    standard_scaler_result_df = standard_scaler(double_vector_value_df)
    standard_scaler_result_df.show(truncate=False)
    # 3.5 区间缩放法
    min_max_scaler_result_df = max_min_scaler(double_vector_value_df)
    min_max_scaler_result_df.show(truncate=False)
    # 3.6 将多列数据合并成为一列, 合并成为的列数据类型为：Vector
    # age, sex, phone, type
    # phone: 130 131 155联通，138 187 188移动，其它是电信的
    person_rdd = sc.parallelize([
        (21.0, 'M', '131xxxxxxxx', 1),
        (22.0, 'F', '138xxxxxxxx', 1),
        (33.0, 'F', '138xxxxxxxx', 1),
        (34.0, 'M', '131xxxxxxxx', 1),
        (45.0, 'M', '187xxxxxxxx', 1),
        (46.0, 'M', '188xxxxxxxx', 0),
        (24.0, 'F', '155xxxxxxxx', 0),
        (18.0, 'M', '130xxxxxxxx', 0),
        (19.0, 'M', '133xxxxxxxx', 0)
    ])


    def fetch_operator(str):
        if str in ('130', '131', '155'):
            return 0.0
        elif str in ('138', '187', '188'):
            return 1.0
        else:
            return 2.0


    def parse_age(age):
        """
        年龄做一个区间化操作
        :param age:
        :return:
        """
        if age < 20:
            return 0.0
        elif age < 30:
            return 1.0
        elif age < 40:
            return 2.0
        else:
            return 3.0


    row_person_rdd = person_rdd.map(lambda t: (t[0], parse_age(t[0]), t[1], t[2][:3], fetch_operator(t[2][:3]), t[3])) \
        .map(lambda t: Row(age=t[0], age2=t[1], sex=t[2], phone=t[3], operator=t[4], p_type=t[5]))
    row_person_df = sqlContext.createDataFrame(row_person_rdd)
    # 开始处理(sex、phone做一个StringIndexer操作)
    sex_string_indexer = StringIndexer(inputCol='sex', outputCol='sex2')
    sex_string_indexer_model = sex_string_indexer.fit(row_person_df)
    row_person_df_tmp01 = sex_string_indexer_model.transform(row_person_df)

    phone_string_indexer = StringIndexer(inputCol='phone', outputCol='phone2')
    phone_string_indexer_model = phone_string_indexer.fit(row_person_df_tmp01)
    row_person_df_tmp02 = phone_string_indexer_model.transform(row_person_df_tmp01)

    # 开始处理(sex2、phone2、operator做一个哑编码操作)
    row_person_df_tmp03 = OneHotEncoder(inputCol='sex2', outputCol='sex_vec').transform(row_person_df_tmp02)
    row_person_df_tmp04 = OneHotEncoder(inputCol='phone2', outputCol='phone_vec').transform(row_person_df_tmp03)
    row_person_df_tmp05 = OneHotEncoder(inputCol='operator', outputCol='operator_vec').transform(row_person_df_tmp04)
    # 合并所有列
    vector_assembler = VectorAssembler(inputCols=['age', 'age2', 'sex_vec', 'phone_vec', 'operator_vec'],
                                       outputCol='features')
    person_feature_result_df = vector_assembler.transform(row_person_df_tmp05)
    person_feature_result_df.show(truncate=False)

    # 四、特征选择
    """
    一般根据特征与特征之间的相关性等指标进行选择，以及根据特征属性和目标属性之间的相关性进行选择
    策略：
    1. 如果多个特征属性之间存在着比较强的相关性的，那么删除特征属性，只留下其中的某一个特征即可
    2. 选择影响目标属性比较大的特征属性，比如和目标属性比较协方差比较大的特征属性留下
    """
    # 4.1 RFormula：直接组合(了解即可)
    rformula = RFormula(formula='p_type ~ age + age2 + sex + phone + operator', \
                        featuresCol='features', labelCol='label')
    rformula_model = rformula.fit(row_person_df)
    rformula_model.transform(row_person_df).show(truncate=False)

    # 五、降维
    """
    一般可以比较形象的理解为：合并特征数据，将多个特征的数据合并成为一个；一般都是通过矩阵的线性变换从而达到合并的效果的/降维的效果的
    """
    pca = PCA(k=2, inputCol='features', outputCol='pca_features')
    pca_model = pca.fit(person_feature_result_df)
    pca_model.transform(person_feature_result_df).show(truncate=False)
