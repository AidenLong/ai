# -- encoding:utf-8 --
import tensorflow as tf

# # 需求一
# # 1. 定义一个变量
# x = tf.Variable(1, dtype=tf.int32, name='v_x')
#
# # 2. 变量的更新
# assign_op = tf.assign(ref=x, value=x + 1)
#
# # 3. 变量初始化操作
# x_init_op = tf.global_variables_initializer()
#
# # 3. 运行
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 变量初始化
#     sess.run(x_init_op)
#
#     # 模拟迭代更新累加器
#     for i in range(5):
#         # 执行更新操作
#         sess.run(assign_op)
#         r_x = sess.run(x)
#         print(r_x)

# # 需求二
# # 1. 定义一个不定形状的变量
# x = tf.Variable(
#     initial_value=[],  # 给定一个空值
#     dtype=tf.float32,
#     trainable=False,
#     validate_shape=False  # 设置为True，表示在变量更新的时候，进行shape的检查，默认为True
# )
#
# # 2. 变量更改
# concat = tf.concat([x, [0.0, 0.0]], axis=0)
# assign_op = tf.assign(x, concat, validate_shape=False)
#
# # 3. 变量初始化操作
# x_init_op = tf.global_variables_initializer()
#
# # 3. 运行
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 变量初始化
#     sess.run(x_init_op)
#
#     # 模拟迭代更新累加器
#     for i in range(5):
#         # 执行更新操作
#         sess.run(assign_op)
#         r_x = sess.run(x)
#         print(r_x)

# 需求三
# 1. 定义一个变量
sum = tf.Variable(1, dtype=tf.int32)
# 2. 定义一个占位符
i = tf.placeholder(dtype=tf.int32)

# 3. 更新操作
tmp_sum = sum * i
# tmp_sum = tf.multiply(sum, i)
assign_op = tf.assign(sum, tmp_sum)
with tf.control_dependencies([assign_op]):
    # 如果需要执行这个代码块中的内容，必须先执行control_dependencies中给定的操作/tensor
    sum = tf.Print(sum, data=[sum, sum.read_value()], message='sum:')

# 4. 变量初始化操作
x_init_op = tf.global_variables_initializer()

# 5. 运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(x_init_op)

    # 模拟迭代更新累加器
    for j in range(1, 6):
        # 执行更新操作
        # sess.run(assign_op, feed_dict={i: j})
        # 通过control_dependencies可以指定依赖关系，这样的话，就不用管内部的更新操作了
        r = sess.run(sum, feed_dict={i: j})

    print("5!={}".format(r))
