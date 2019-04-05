import tensorflow as tf
from mobileNet_layers import *

class MobileNet(object):

    def __init__(self, inputs, num_classes=43, is_training=True,
                 width_multiplier=1, scope="MobileNet"):
        """
        The implement of MobileNet(ref:https://arxiv.org/abs/1704.04861)
        :inputs: 4-D Tensor of [batch_size, height, width, channels]
        :num_classes: number of classes
        :is_training: Boolean, whether or not the model is training
        :width_multiplier: float, controls the size of model
        :scope: Optional scope for variables
        """

        self.inputs = inputs # 输入张量
        self.num_classes = num_classes # 类别数
        self.is_training = is_training # 是否训练
        self.width_multiplier = width_multiplier

        # 构建模型
        with tf.variable_scope(scope):
            # 输入 48*48*3
            # conv1  3*3卷积，数量round(32 * width_multiplier)
            net = conv2d(inputs, "conv_1", round(32 * width_multiplier), filter_size=3,strides=2)  # ->[N, 24, 24, 32]
            net = tf.nn.relu(bacthnorm(net, "conv_1/bn", is_training=self.is_training))
            # 13个深度可分离卷积模块
            net = depthwise_separable_conv2d(net, 64, self.width_multiplier, "ds_conv_2",self.is_training) # ->[N, 24, 24, 64]
            # downsample=True 进行下采样，步长为2，尺寸减半
            net = depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_3",self.is_training, downsample=True) # ->[N, 12, 12, 128]
            net = depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_4",self.is_training) # ->[N, 12, 12, 128]
            
            net = depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_5", self.is_training,downsample=True) # ->[N, 6, 6, 256]
            net = depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_6",self.is_training) # ->[N, 6, 6, 256]

            net = avg_pool(net, 6, "avg_pool_15")
            
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            """
            tf.squeeze 返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
            axis可以用来指定要删掉的为1的维度，此处要注意指定的维度必须确保其是1，否则会报错
            """
            self.logits = fc(net, self.num_classes, "fc_16")
            self.predictions = tf.nn.softmax(self.logits)

if __name__ == "__main__":

    # 随机测试数据
    inputs = tf.random_normal(shape=[100, 48, 48, 3])

    mobileNet = MobileNet(inputs)
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(mobileNet.predictions)
        print(pred.shape)
