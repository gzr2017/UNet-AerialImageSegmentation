from src.model import *


def pixel_wise_softmax(output_map):
    with tf.name_scope('pixel_wise_softmax'):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


class NeuralNet(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32,
                                shape=[None, None, None, None],
                                name='x')
        self.y = tf.placeholder(tf.float32,
                                shape=[None, None, None, None],
                                name='y')
        self.output_map = build_net(self.x)
        self.cost, self.cross_entropy = self._get_cost('class_balanced_sigmoid_cross_entropy')
        with tf.name_scope('results'):
            max_pos_y = tf.argmax(self.y, axis=3)
            max_pos_output = tf.argmax(self.output_map, axis=3)
            flat_max_pos_y = tf.reshape(max_pos_y, shape=[-1])
            flat_max_pos_output = tf.reshape(max_pos_output, shape=[-1])
            confusion_matrix = tf.confusion_matrix(labels=flat_max_pos_y, predictions=flat_max_pos_output)
            TP = confusion_matrix[0, 0]
            FN = confusion_matrix[0, 1]
            FP = confusion_matrix[1, 0]
            TN = confusion_matrix[1, 1]
            self.accuracy = (TP + TN) / (TP + TN + FP + FN)
            self.precision = TP / (TP + FP)
            self.recall = TP / (TP + FN)
            self.f1_score = (2 * self.precision * self.recall) / (self.precision + self.recall)

    def _get_cost(self, entropy_weight):
        flat_y = tf.reshape(self.y, [-1, n_class])
        flat_output_map = tf.reshape(self.output_map, [-1, n_class])
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=flat_output_map, labels=flat_y))
        if entropy_weight == 'softmax_cross_entropy':
            return cross_entropy, cross_entropy
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
            return cost, cross_entropy
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
