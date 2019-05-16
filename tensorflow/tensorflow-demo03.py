# -- encoding:utf-8 --
import tensorflow as tf


# with tf.device('/cpu:0'):
#     # 这个代码块中定义的操作，会在tf.device给定的设备上运行
#     # 有一些操作，是不会再GPU上运行的（一定要注意）
#     # 如果安装的tensorflow cpu版本，没法指定运行环境的
#     a = tf.Variable([1, 2, 3], dtype=tf.int32, name='a')
#     b = tf.constant(2, dtype=tf.int32, name='b')
#     c = tf.add(a, b, name='ab')
#
# with tf.device('/gpu:0'):
#     # 这个代码块中定义的操作，会在tf.device给定的设备上运行
#     # 有一些操作，是不会再GPU上运行的（一定要注意）
#     # 如果按照的tensorflow cpu版本，没法指定运行环境的
#     d = tf.Variable([2, 8, 13], dtype=tf.int32, name='d')
#     e = tf.constant(2, dtype=tf.int32, name='e')
#     f = d + e
#
# g = c + f
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 初始化
#     tf.global_variables_initializer().run()
#     # 执行结果
#     # print(g.eval())
#     print(c.eval())

# 方式一
# def my_func(x):
#     w1 = tf.Variable(tf.random_normal([1]))[0]
#     b1 = tf.Variable(tf.random_normal([1]))[0]
#     r1 = w1 * x + b1
#
#     w2 = tf.Variable(tf.random_normal([1]))[0]
#     b2 = tf.Variable(tf.random_normal([1]))[0]
#     r2 = w2 * r1 + b2
#
#     return r1, w1, b1, r2, w2, b2
#
#
# # 下面两行代码还是属于图的构建
# x = tf.constant(3, dtype=tf.float32)
# r = my_func(x)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 初始化
#     tf.global_variables_initializer().run()
#     # 执行结果
#     print(sess.run(r))

# 方式二
# def my_func(x):
#     # initializer：初始化器
#     # w = tf.Variable(tf.random_normal([1]), name='w')[0]
#     # b = tf.Variable(tf.random_normal([1]), name='b')[0]
#     w = tf.get_variable(name='w', shape=[1], initializer=tf.random_normal_initializer())[0]
#     b = tf.get_variable(name='b', shape=[1], initializer=tf.random_normal_initializer())[0]
#     r = w * x + b
#
#     return r, w, b
#
#
# def func(x):
#     with tf.variable_scope('op1', reuse=tf.AUTO_REUSE):
#         r1 = my_func(x)
#     with tf.variable_scope('op2', reuse=tf.AUTO_REUSE):
#         r2 = my_func(r1[0])
#     return r1, r2
#
#
# # 下面两行代码还是属于图的构建
# x1 = tf.constant(3, dtype=tf.float32, name='x1')
# x2 = tf.constant(4, dtype=tf.float32, name='x2')
# with tf.variable_scope('func1'):
#     r1 = func(x1)
# with tf.variable_scope('func2'):
#     r2 = func(x2)

# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 初始化
#     tf.global_variables_initializer().run()
#     # 执行结果
#     print(sess.run([r1, r2]))


# variable_scope嵌套
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    with tf.variable_scope('foo', initializer=tf.constant_initializer(4.0)) as foo:
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(3.0))
        with tf.variable_scope('bar'):
            l = tf.get_variable("l", [1])

            with tf.variable_scope(foo):
                h = tf.get_variable('h', [1])
                g = v + w + l + h

    with tf.variable_scope('abc'):
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(5.0))
        b = a + g

    sess.run(tf.global_variables_initializer())
    print("{},{}".format(v.name, v.eval()))
    print("{},{}".format(w.name, w.eval()))
    print("{},{}".format(l.name, l.eval()))
    print("{},{}".format(h.name, h.eval()))
    print("{},{}".format(g.name, g.eval()))
    print("{},{}".format(a.name, a.eval()))
    print("{},{}".format(b.name, b.eval()))


# # 可视化
# with tf.device("/cpu:0"):
#     with tf.variable_scope("foo"):
#         x_init1 = tf.get_variable('init_x', [10], tf.float32, initializer=tf.random_normal_initializer())[0]
#         x = tf.Variable(initial_value=x_init1, name='x')
#         y = tf.placeholder(dtype=tf.float32, name='y')
#         z = x + y
#
#     # update x
#     assign_op = tf.assign(x, x + 1)
#     with tf.control_dependencies([assign_op]):
#         with tf.device('/cpu:0'):
#             out = x * y
#
# # with tf.device('/cpu:0'):
#     with tf.variable_scope("bar"):
#         a = tf.constant(3.0) + 4.0
#     w = z * a
#
# # 开始记录信息(需要展示的信息的输出)
# tf.summary.scalar('scalar_init_x', x_init1)
# tf.summary.scalar(name='scalar_x', tensor=x)
# tf.summary.scalar('scalar_y', y)
# tf.summary.scalar('scalar_z', z)
# tf.summary.scalar('scala_w', w)
# tf.summary.scalar('scala_out', out)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # merge all summary
#     merged_summary = tf.summary.merge_all()
#     # 得到输出到文件的对象
#     writer = tf.summary.FileWriter('./result', sess.graph)
#
#     # 初始化
#     sess.run(tf.global_variables_initializer())
#     # print
#     for i in range(1, 5):
#         summary, r_out, r_x, r_w = sess.run([merged_summary, out, x, w], feed_dict={y: i})
#         writer.add_summary(summary, i)
#         print("{},{},{}".format(r_out, r_x, r_w))
#
#     # 关闭操作
#     writer.close()

# # 模型保存
# v1 = tf.Variable(tf.constant(3.0), name='v1')
# v2 = tf.Variable(tf.constant(4.0), name='v2')
# result = v1 + v2
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(result)
#     # 模型保存到model文件夹下，文件前缀为：model.ckpt
#     saver.save(sess, './model/model.ckpt')

# 模型的提取(完整提取：需要完整恢复保存之前的数据格式)
v1 = tf.Variable(tf.constant(1.0), name='v1')
v2 = tf.Variable(tf.constant(4.0), name='v2')
result = v1 + v2

saver = tf.train.Saver()
with tf.Session() as sess:
    # 会从对应的文件夹中加载变量、图等相关信息
    saver.restore(sess, './model/model.ckpt')
    print(sess.run([result]))

# 直接加载图，不需要定义变量了
saver = tf.train.import_meta_graph('./model/model.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))

# 模型的提取(给定映射关系)
a = tf.Variable(tf.constant(1.0), name='a')
b = tf.Variable(tf.constant(2.0), name='b')
result = a + b

saver = tf.train.Saver({"v1": a, "v2": b})
with tf.Session() as sess:
    # 会从对应的文件夹中加载变量、图等相关信息
    saver.restore(sess, './model/model.ckpt')
    print(sess.run([result]))
    # writer = tf.summary.FileWriter('./result', sess.graph)
    # writer.close()
