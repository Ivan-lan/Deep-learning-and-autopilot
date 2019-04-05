import tensorflow as tf
from tensorflow.python.training import moving_averages

UPDATE_OPS_COLLECTION = "_update_ops_"

# 创建变量
def create_variable(name, shape, initializer,dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,initializer=initializer, trainable=trainable)

# BN层
def bacthnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):

    inputs_shape = inputs.get_shape().as_list()
    params_shape = inputs_shape[-1:]
    axis = list(range(len(inputs_shape) - 1))

    with tf.variable_scope(scope):
        beta = create_variable("beta", params_shape,initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_variable("moving_mean", params_shape,
                                       initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = create_variable("moving_variance", params_shape,
                                       initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=axis) # 计算均值方差
        update_move_mean = moving_averages.assign_moving_average(moving_mean,mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance, variance, decay=momentum)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance

    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)


# depthwise卷积层
def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,in_channels, channel_multiplier],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))

    return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, strides, strides, 1],
                                  padding="SAME", rate=[1, 1]) # 输出通道变成了in_channels * channel_multiplier
    # 参考 https://blog.csdn.net/mao_xiao_feng/article/details/78003476

# 普通卷积层
def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,in_channels, num_filters],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.conv2d(inputs, filter, strides=[1, strides, strides, 1],padding="SAME")

# 平均池化层
def avg_pool(inputs, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1],
                              strides=[1, pool_size, pool_size, 1], padding="VALID")

# 全连接层
def fc(inputs, n_out, scope, use_bias=True):
    inputs_shape = inputs.get_shape().as_list()
    n_in = inputs_shape[-1]
    with tf.variable_scope(scope):
        weight = create_variable("weight", shape=[n_in, n_out],
                                  initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:
            bias = create_variable("bias", shape=[n_out,],initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs, weight, bias)
        return tf.matmul(inputs, weight)

# 深度可分离卷积模块
def depthwise_separable_conv2d(inputs, num_filters, width_multiplier,
                               scope, is_training, downsample=False):
    """depthwise separable convolution 2D function"""
    num_filters = round(num_filters * width_multiplier)
    strides = 2 if downsample else 1 # 是否下采样

    with tf.variable_scope(scope):
        # depthwise卷积层
        dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
        # BN层
        bn = bacthnorm(dw_conv, "dw_bn", is_training=is_training)
        # relu层
        relu = tf.nn.relu(bn)
        # pointwise卷积层
        pw_conv = conv2d(relu, "pointwise_conv", num_filters)
        # BN层
        bn = bacthnorm(pw_conv, "pw_bn", is_training=is_training)
        return tf.nn.relu(bn) # relu层

"""
注释：

tf.nn.batch_normalization(x,mean,variance,offset,scale,variance_epsilon,name=None)
x:input
mean:样本均值
variance:样本方差
offset:样本偏移(相加一个转化值)
scale:缩放（默认为1）
variance_epsilon:为了避免分母为0，添加的一个极小值
原文：https://blog.csdn.net/qq_37972530/article/details/82749624 

BN在实际中，由于mean和variance是和batch内的数据有关的，因此需要注意训练过程和预测过程中，mean和variance无法使用相同的数据。
需要一个trick，即moving_average，在训练的过程中，通过每个step得到的mean和variance，叠加计算对应的moving_average（滑动平均），
并最终保存下来以便在inference的过程中使用。

assign_moving_average(variable, value, decay, zero_debias=True, name=None)
其实内部计算比较简单，公式表达如下：
variable = variable * decay + value * (1 - decay)
变换一下：
variable = variable - (1 - decay) * (variable - value)
减号后面的项就是moving_average的更新delta了。

链接：https://www.jianshu.com/p/7ce4e709fe7d

"""