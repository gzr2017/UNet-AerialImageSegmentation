from src.model import *


def pixel_wise_softmax(output_map):
    with tf.name_scope('pixel_wise_softmax'):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


class NeuralNet(object):
    def __init__(self):
        n_class = len(list(color_class_dict.keys()))
        self.x = tf.placeholder(tf.float32,
                                shape=[None, None, None, None],
                                name='x')
        self.y = tf.placeholder(tf.float32,
                                shape=[None, None, None, None],
                                name='y')
        self.output_map = build_net(self.x)
        self.cost = self._get_cost('softmax_cross_entropy', n_class)
        with tf.name_scope('results'):
            flat_y = tf.reshape(self.y, [-1, n_class])
            flat_output_map = tf.reshape(self.output_map, [-1, n_class])
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=flat_output_map, labels=flat_y))
            prediction = pixel_wise_softmax(self.output_map)
            self.correct_pred = tf.equal(tf.argmax(prediction, 3),
                                         tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, entropy_weight, n_class):
        flat_y = tf.reshape(self.y, [-1, n_class])
        flat_output_map = tf.reshape(self.output_map, [-1, n_class])
        if entropy_weight == 'softmax_cross_entropy':
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=flat_output_map, labels=flat_y))
            return cost
        elif entropy_weight == 'class_balanced_sigmoid_cross_entropy':
            if n_class != 2:
                raise ValueError('该损失函数只能计算二分类问题')

            count_neg = tf.reduce_sum(1. - flat_y)
            count_pos = tf.reduce_sum(flat_y)
            beta = count_neg / (count_neg + count_pos)
            pos_weight = beta / (1 - beta)
            cost = tf.nn.weighted_cross_entropy_with_logits(
                logits=flat_output_map, targets=flat_y, pos_weight=pos_weight)
            cost = tf.reduce_mean(cost * (1 - beta))
            return cost
        else:
            raise ValueError('未明的损失函数')

    def predict(self, x, model_save_path):
        x = np.reshape(x, newshape=[1, x.shape[0], x.shape[1], -1])
        with tf.Session() as sess:
            self.restore(sess, model_save_path)
            output_map = sess.run(self.output_map, feed_dict={self.x: x})
            return class_to_color(output_map_to_class(output_map[0]))

    @staticmethod
    def save(sess, model_save_path):
        saver = tf.train.Saver()
        saver.save(sess, model_save_path)

    @staticmethod
    def restore(sess, model_save_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)
