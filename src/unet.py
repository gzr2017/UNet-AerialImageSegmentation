from src.model import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class UNet(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32,
                                shape=[None, None, None, IMG_CHANNEL],
                                name='x')
        self.y = tf.placeholder(tf.float32,
                                shape=[None, None, None, N_CLASS],
                                name='y')
        self.output_map = build_unet(self.x)
        self.cost = self._get_cost()
        with tf.name_scope('results'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=tf.reshape(self.output_map, [-1, N_CLASS]),
                    labels=tf.reshape(self.y, [-1, N_CLASS])))
            self.correct_pred = tf.equal(tf.argmax(self.output_map, 3),
                                         tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self):
        with tf.name_scope('cost'):
            flat_y = tf.reshape(self.y, [-1, N_CLASS])
            flat_output_map = tf.reshape(self.output_map, [-1, N_CLASS])
            if HOW_TO_CAL_COST == 'cross_entropy':
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=flat_output_map, labels=flat_y))
                return loss
            elif HOW_TO_CAL_COST == 'weighted_cross_entropy':  # 带权重的交叉熵
                weight_map = tf.multiply(flat_y, CLASS_WEIGHT)
                weight_map = tf.reduce_sum(weight_map, axis=1)
                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_y, logits=flat_output_map)
                weighted_loss = tf.multiply(loss_map, weight_map)
                return tf.reduce_mean(weighted_loss)
            elif HOW_TO_CAL_COST == 'sigmoid_cross_entropy_balanced':
                count_neg = tf.reduce_sum(1. - self.y)
                count_pos = tf.reduce_sum(self.y)
                beta = count_neg / (count_neg + count_pos)
                pos_weight = beta / (1 - beta)
                cost = tf.nn.weighted_cross_entropy_with_logits(logits=flat_output_map, targets=flat_y,
                                                                post_weight=pos_weight)
                cost = tf.reduce_mean(cost * (1 - beta))
                return cost
            else:
                raise ValueError('未知损失函数计算方法%s' % HOW_TO_CAL_COST)

    def predict(self, image, model_path, save_path):
        test_x = np.array(Image.open(image))
        test_x = np.reshape(test_x,
                            newshape=[1, test_x.shape[0], test_x.shape[1], -1])
        with tf.Session() as sess:
            self.restore(sess, model_path)
            output_map = sess.run(self.output_map,
                                  feed_dict={
                                      self.x: test_x,
                                  })
            save_name = path.basename(image).split('.')[-2]
            fake_label = np.zeros(shape=output_map.shape, dtype=tf.uint8)
            prediction = output_class(output_map)
            class_to_color(test_x, fake_label, prediction, save_path, save_name)

    @staticmethod
    def save(sess, save_path, save_name):
        saver = tf.train.Saver()
        saver.save(sess, path.join(save_path, save_name))

    @staticmethod
    def restore(sess, save_path):
        saver = tf.train.Saver()
        saver.restore(sess, path.join(save_path, RESTORE))
