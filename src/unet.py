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
        self.output_map, self.variables = build_unet(self.x)
        self.cost = self._get_cost()
        self.gradients_node = tf.gradients(self.cost, self.variables)
        with tf.name_scope('results'):
            # 交叉熵
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=tf.reshape(self.output_map, [-1, N_CLASS]),
                    labels=tf.reshape(self.y, [-1, N_CLASS])))
            self.correct_pred = tf.equal(tf.argmax(self.output_map, 3),
                                         tf.argmax(self.y, 3))
            # 准确率
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self):
        with tf.name_scope('cost'):
            flat_y = tf.reshape(self.y, [-1, N_CLASS])
            flat_output_map = tf.reshape(self.output_map, [-1, N_CLASS])
            if HOW_TO_CAL_COST == 'cross_entropy':
                return tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_y, logits=flat_output_map)
            elif HOW_TO_CAL_COST == 'weighted_cross_entropy':  # 带权重的交叉熵
                weight_map = tf.multiply(flat_y, CLASS_WEIGHT)
                weight_map = tf.reduce_sum(weight_map, axis=1)
                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_y, logits=flat_output_map)
                weighted_loss = tf.multiply(loss_map, weight_map)
                return tf.reduce_mean(weighted_loss)
            elif HOW_TO_CAL_COST == 'dice_coefficient':
                pass
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
