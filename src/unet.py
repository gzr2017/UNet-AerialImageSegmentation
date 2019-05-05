from src.model import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def output_class(prediction):
    most_possible_position = np.argmax(prediction, 3)
    classed_prediction = np.zeros(shape=prediction.shape)
    [batch_size, rows, cols] = most_possible_position.shape
    for h in range(batch_size):
        for i in range(rows):
            for j in range(cols):
                classed_prediction[h, i, j,
                                   most_possible_position[h, i, j]] = 1
    return classed_prediction


def class_to_color(classed_prediction, save_path, save_name):
    # TODO: 增加batch_size不为1的情况
    reverse_COLOR_CLASS_DICT = dict(
        zip(COLOR_CLASS_DICT.values(), COLOR_CLASS_DICT.keys()))
    [batch_size, rows, cols, _] = classed_prediction.shape
    for h in range(batch_size):
        color_value = np.zeros(
            (rows, cols, 1), dtype=np.uint8)
        # (rows, cols, len(reverse_COLOR_CLASS_DICT.values()[0])), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                for k in range(N_CLASS):
                    if classed_prediction[h][i][j][k] == 1:
                        color_value[i][j][0:IMG_CHANNEL] = reverse_COLOR_CLASS_DICT[k]
                        break
        if IMG_CHANNEL == 1:
            color_value = np.reshape(color_value, [rows, cols])
        real_out_predict_image = Image.fromarray(color_value.astype('uint8'))
        real_out_predict_image.save(
            path.join(save_path, save_name + '_'+str(h) + '.tif'))


class UNet(object):
    def __init__(self, cost_kwargs={}, **kwargs):
        self.x = tf.placeholder(
            tf.float32, shape=[None, None, None, IMG_CHANNEL], name='x')
        self.y = tf.placeholder(
            tf.float32, shape=[None, None, None, N_CLASS], name='y')
        self.output_map, self.variables = build_unet(self.x)
        self.cost = self._get_cost(self.output_map, cost_kwargs)
        self.gradients_node = tf.gradients(self.cost, self.variables)
        with tf.name_scope('results'):
            # 交叉熵
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(self.output_map, [-1, N_CLASS]),
                                                           labels=tf.reshape(self.y, [-1, N_CLASS])))
            self.correct_pred = tf.equal(
                tf.argmax(self.output_map, 3), tf.argmax(self.y, 3))
            # 准确率
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, output_map, cost_kwargs):
        with tf.name_scope('cost'):
            if HOW_TO_CAL_COST == 'cross_entropy':
                flat_logits = tf.reshape(output_map, [-1, N_CLASS])
                flat_labels = tf.reshape(self.y, [-1, N_CLASS])
                class_weights = cost_kwargs.pop('class_weights', None)
                if class_weights is not None:
                    class_weights = tf.constant(
                        np.array(class_weights, dtype=np.float32))
                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)
                    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=flat_logits, labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)
                    loss = tf.reduce_mean(weighted_loss)
                else:
                    loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels))
            elif HOW_TO_CAL_COST == 'dice_coefficient':
                pass
            else:
                raise ValueError('未知损失函数计算方法%s' % HOW_TO_CAL_COST)
            regularizer = cost_kwargs.pop('regularizer', None)
            if regularizer is not None:
                regularizers = sum(
                    [tf.nn.l2_loss(variable) for variable in self.variables])
                loss += (regularizer * regularizers)
            return loss

    def predict(self, image, model_path, save_path):
        test_x = np.array(Image.open(image))
        test_x = np.reshape(
            test_x, newshape=[1, test_x.shape[0], test_x.shape[1], -1])
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            self.restore(sess, model_path)
            output_map = sess.run(self.output_map,
                                  feed_dict={
                                      self.x: test_x,
                                  })
            save_name = path.basename(image).split('.')[-2]
            prediction = output_class(output_map)
            class_to_color(prediction, save_path, save_name)

    @staticmethod
    def save(sess, save_path, save_name):
        saver = tf.train.Saver()
        saver.save(sess, path.join(save_path, save_name))

    @staticmethod
    def restore(sess, save_path):
        # 并没有名为 unet.ckpt 的实体文件。它是为检查点创建的文件名的前缀。用户仅与前缀（而非检查点实体文件）互动。
        saver = tf.train.Saver()
        saver.restore(sess, path.join(save_path, RESTORE))


if __name__ == '__main__':
    prediction = np.load(
        'D:\GitHub\/UNet-AerialImageSegmentation\/fake_data\/train\split\label_classed\/austin1_0.npy')
    prediction = np.reshape(
        prediction, newshape=[-1, prediction.shape[0], prediction.shape[1], prediction.shape[2]])
    prediction = output_class(prediction)
    class_to_color(prediction, '.', 'test')
