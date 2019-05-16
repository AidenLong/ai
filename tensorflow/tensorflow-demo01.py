# -- encoding:utf-8 --

import tensorflow as tf

# # 1. 定义常量矩阵a和矩阵b
# # name属性只是给定这个操作一个名称而已
# a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32, name='a')
# print(type(a))
# b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2], name='b')
#
# # 2. 以a和b作为输入，进行矩阵的乘法操作
# c = tf.matmul(a, b, name='matmul')
# print(type(c))
#
# # 3. 以a和c作为输入，进行矩阵的相加操作
# g = tf.add(a, c, name='add')
# print(type(g))
# print(g)
#
# # 4. 添加减法
# h = tf.subtract(b, a, name='b-a')
# l = tf.matmul(h, c)
# r = tf.add(g, l)
#
# print("变量a是否在默认图中:{}".format(a.graph is tf.get_default_graph()))

# # 使用新的构建的图
# graph = tf.Graph()
# with graph.as_default():
#     # 此时在这个代码块中，使用的就是新的定义的图graph(相当于把默认图换成了graph)
#     d = tf.constant(5.0, name='d')
#     print("变量d是否在新图graph中:{}".format(d.graph is graph))
#
# with tf.Graph().as_default() as g2:
#     e = tf.constant(6.0)
#     print("变量e是否在新图g2中：{}".format(e.graph is g2))
#
# # 这段代码是错误的用法，记住：不能使用两个图中的变量进行操作，只能对同一个图中的变量对象（张量）进行操作(op)
# # f = tf.add(d, e)


# 会话构建&启动(默认情况下（不给定Session的graph参数的情况下），创建的Session属于默认的图)
# sess = tf.Session()
# print(sess)
#
# # 调用sess的run方法来执行矩阵的乘法，得到c的结果值（所以将c作为参数传递进去）
# # 不需要考虑图中间的运算，在运行的时候只需要关注最终结果对应的对象以及所需要的输入数据值
# # 只需要传递进去所需要得到的结果对象，会自动的根据图中的依赖关系触发所有相关的OP操作的执行
# # 如果op之间没有依赖关系，tensorflow底层会并行的执行op(有资源) --> 自动进行
# # 如果传递的fetches是一个列表，那么返回值是一个list集合
# # fetches：表示获取那个op操作的结果值
# result = sess.run(fetches=[r, c])
# print("type:{}, value:\n{}".format(type(result), result))
#
# # 会话关闭
# sess.close()
#
# # 当一个会话关闭后，不能再使用了，所以下面两行代码错误
# # result2 = sess.run(c)
# # print(result2)
#
# # 使用with语句块，会在with语句块执行完成后，自动的关闭session
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess2:
#     print(sess2)
#     # 获取张量c的结果： 通过Session的run方法获取
#     print("sess2 run:{}".format(sess2.run(c)))
#     # 获取张量r的结果：通过张量对象的eval方法获取，和Session的run方法一致
#     print("c eval:{}".format(r.eval()))
#
# # 交互式会话构建
# sess3 = tf.InteractiveSession()
# print(r.eval())

# # 1. 定义一个变量，必须给定初始值(图的构建，没有运行)
# a = tf.Variable(initial_value=3.0, dtype=tf.float32)
#
# # 2. 定义一个张量
# b = tf.constant(value=2.0, dtype=tf.float32)
# c = tf.add(a, b)
#
# # 3. 进行初始化操作（推荐：使用全局所有变量初始化API）
# # 相当于在图中加入一个初始化全局变量的操作
# init_op = tf.global_variables_initializer()
# print(type(init_op))
#
# # 3. 图的运行
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 运行init op进行变量初始化，一定要放到所有运行操作之前
#     sess.run(init_op)
#     # init_op.run() # 这行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())是哪个，底层使用默认session来运行
#     # 获取操作的结果
#     print("result:{}".format(sess.run(c)))
#     print("result:{}".format(c.eval()))

# # 1. 定义变量，常量
# w1 = tf.Variable(tf.random_normal(shape=[10], stddev=0.5, seed=28, dtype=tf.float32), name='w1')
# a = tf.constant(value=2.0, dtype=tf.float32)
# w2 = tf.Variable(w1.initialized_value() * a, name='w2')
#
# # 3. 进行初始化操作（推荐：使用全局所有变量初始化API）
# # 相当于在图中加入一个初始化全局变量的操作
# init_op = tf.global_variables_initializer()
# print(type(init_op))
#
# # 3. 图的运行
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 运行init op进行变量初始化，一定要放到所有运行操作之前
#     sess.run(init_op)
#     # init_op.run() # 这行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())是哪个，底层使用默认session来运行
#     # 获取操作的结果
#     print("result:{}".format(sess.run(w1)))
#     print("result:{}".format(w2.eval()))

# # 构建一个矩阵的乘法，但是矩阵在运行的时候给定
# m1 = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='placeholder_1')
# m2 = tf.placeholder(dtype=tf.float32, shape=[3, 2], name='placeholder_2')
# m3 = tf.matmul(m1, m2)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     print("result:\n{}".format(
#         sess.run(fetches=m3, feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})))
#     print("result:\n{}".format(m3.eval(feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})))

