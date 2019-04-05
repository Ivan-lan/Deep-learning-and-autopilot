import tensorflow as tf
from mobileNet_layers import *

class MobileNet(object):

    def __init__(self, inputs, num_classes=1000, is_training=True,
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
            # conv1  3*3卷积，数量round(32 * width_multiplier)
            net = conv2d(inputs, "conv_1", round(32 * width_multiplier), filter_size=3,strides=2)  # ->[N, 112, 112, 32]
            net = tf.nn.relu(bacthnorm(net, "conv_1/bn", is_training=self.is_training))
            # 13个深度可分离卷积模块
            net = depthwise_separable_conv2d(net, 64, self.width_multiplier, "ds_conv_2",self.is_training) # ->[N, 112, 112, 64]
            
            net = depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_3",self.is_training, downsample=True) # ->[N, 56, 56, 128]
            net = depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_4",self.is_training) # ->[N, 56, 56, 128]
            
            net = depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_5", self.is_training,downsample=True) # ->[N, 28, 28, 256]
            net = depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_6",self.is_training) # ->[N, 28, 28, 256]
            
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_7",self.is_training, downsample=True) # ->[N, 14, 14, 512]
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_8",self.is_training) # ->[N, 14, 14, 512]
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier,"ds_conv_9",self.is_training)  # ->[N, 14, 14, 512]
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier,"ds_conv_10",self.is_training)  # ->[N, 14, 14, 512]
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier,"ds_conv_11",self.is_training)  # ->[N, 14, 14, 512]
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier,"ds_conv_12",self.is_training)  # ->[N, 14, 14, 512]
            
            net = depthwise_separable_conv2d(net, 1024, self.width_multiplier,"ds_conv_13", self.is_training,downsample=True) # ->[N, 7, 7, 1024]
            net = depthwise_separable_conv2d(net, 1024, self.width_multiplier,"ds_conv_14",self.is_training) # ->[N, 7, 7, 1024]

            net = avg_pool(net, 7, "avg_pool_15")
            
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            self.logits = fc(net, self.num_classes, "fc_16")
            self.predictions = tf.nn.softmax(self.logits)

if __name__ == "__main__":

    # 随机测试数据
    inputs = tf.random_normal(shape=[4, 224, 224, 3])

    mobileNet = MobileNet(inputs)
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(mobileNet.predictions)
        print(pred.shape)
