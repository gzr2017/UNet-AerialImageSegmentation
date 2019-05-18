from collections import OrderedDict
from src.data import *

N_CLASS = 2
IMG_CHANNEL = 3
FILTER_SIZE = 3
POOL_SIZE = 2
LAYERS = 4
FEATURES_ROOT = 64


# 生成卷积核
def weight_variable(shape, name):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


# 生成偏差


def bias_variable(shape, name):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


# 下采样层
def convoluting(x, w, b):
    with tf.name_scope('convoluting'):
        convoluting_layer = tf.nn.conv2d(x,
                                         w,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
        convoluting_layer = tf.nn.bias_add(convoluting_layer, b)
        return convoluting_layer


# 上采样层
def deconvoluting(x, w):
    with tf.name_scope('deconvoluting'):
        output_shape = tf.stack([
            tf.shape(x)[0],
            tf.shape(x)[1] * 2,
            tf.shape(x)[2] * 2,
            tf.shape(x)[3] // 2
        ])
        return tf.nn.conv2d_transpose(x,
                                      w,
                                      output_shape,
                                      strides=[1, POOL_SIZE, POOL_SIZE, 1],
                                      padding='SAME')


# 最大池化层
def maxpooling(x):
    return tf.nn.max_pool(x,
                          ksize=[1, POOL_SIZE, POOL_SIZE, 1],
                          strides=[1, POOL_SIZE, POOL_SIZE, 1],
                          padding='SAME')


def crop_and_concat(x1, x2):
    # TODO: 上卷积过程中，卷积层的尺寸有可能大于下卷积层的尺寸导致offsets为负数
    # 如果我写完了分割工具的话这个TODO可以不DO
    with tf.name_scope('crop_and_concat'):
        offsets = [
            0, (tf.shape(x1)[1] - tf.shape(x2)[1]) // 2,
            (tf.shape(x1)[2] - tf.shape(x2)[2]) // 2, 0
        ]
        size = [-1, tf.shape(x2)[1], tf.shape(x2)[2], -1]
        x1 = tf.slice(x1, offsets, size)
        return tf.concat([x1, x2], 3)


# 生成UNet
def build_net(x):
    in_node = x
    pools = OrderedDict()
    deconvs = OrderedDict()
    dw_h_convs = OrderedDict()  # 存放要与后面上采样做concatenate的卷积层
    up_h_convs = OrderedDict()
    logging.info('开始搭建UNet！')
    for layer in range(0, LAYERS + 1):
        with tf.name_scope('down_conv_{}'.format(str(layer))):
            features = 2**layer * FEATURES_ROOT
            if layer == 0:  # 如果是输入层
                w1 = weight_variable(
                    [FILTER_SIZE, FILTER_SIZE, IMG_CHANNEL, features],
                    name='down_conv_{}_w1'.format(str(layer)))
            else:
                w1 = weight_variable(
                    [FILTER_SIZE, FILTER_SIZE, features // 2, features],
                    name='down_conv_{}_w1'.format(str(layer)))
            w2 = weight_variable(
                [FILTER_SIZE, FILTER_SIZE, features, features],
                name='down_conv_{}_w2'.format(str(layer)))
            b1 = bias_variable([features],
                               name='down_conv_{}_b1'.format(str(layer)))
            b2 = bias_variable([features],
                               name='down_conv_{}_bq'.format(str(layer)))
            conv1 = convoluting(in_node, w1, b1)
            conv2 = convoluting(tf.nn.relu(conv1), w2, b2)
            dw_h_convs[layer] = tf.nn.relu(conv2)
            if layer < LAYERS + 1:
                pools[layer] = maxpooling(dw_h_convs[layer])
                in_node = pools[layer]
    in_node = dw_h_convs[LAYERS]
    for layer in range(LAYERS, 0, -1):
        with tf.name_scope('up_conv_{}'.format(str(layer))):
            features = 2**layer * FEATURES_ROOT
            wd = weight_variable(
                [POOL_SIZE, POOL_SIZE, features // 2, features],
                name='up_conv_{}_wd'.format(str(layer)))
            bd = bias_variable([features // 2],
                               name='up_conv_{}_bd'.format(str(layer)))
            h_deconv = tf.nn.relu(
                tf.add(deconvoluting(in_node, wd),
                       bd,
                       name='up_conv_{}_h_deconv'.format(str(layer))))
            deconvs[layer] = crop_and_concat(dw_h_convs[layer - 1], h_deconv)
            w1 = weight_variable(
                [FILTER_SIZE, FILTER_SIZE, features, features // 2],
                name='up_conv_{}_w1'.format(str(layer)))
            w2 = weight_variable(
                [FILTER_SIZE, FILTER_SIZE, features // 2, features // 2],
                name='up_conv_{}_w2'.format(str(layer)))
            b1 = bias_variable([features // 2],
                               name='up_conv_{}_b1'.format(str(layer)))
            b2 = bias_variable([features // 2],
                               name='up_conv_{}_b2'.format(str(layer)))
            conv1 = convoluting(deconvs[layer], w1, b1)
            conv2 = convoluting(tf.nn.relu(conv1), w2, b2)
            up_h_convs[layer] = tf.nn.relu(conv2)
            in_node = up_h_convs[layer]
    with tf.name_scope('output_map'):
        w = weight_variable([1, 1, FEATURES_ROOT, N_CLASS], name='w')
        b = bias_variable([N_CLASS], name='b')
        conv = convoluting(in_node, w, b)
        output_map = tf.nn.relu(conv)
        up_h_convs['out'] = output_map
    logging.info('UNet搭建完毕！')
    return output_map
