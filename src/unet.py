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
                        color_value[i][j][0:1] = reverse_COLOR_CLASS_DICT[k]
                        break
#TODO: dict的key长度为1
        color_value = np.reshape(color_value, [rows, cols])
        real_out_predict_image = Image.fromarray(color_value.astype('uint8'))
        real_out_predict_image.save(
            path.join(save_path, save_name + '_'+str(h) + '.tif'))


def pixel_wise_softmax(output_map):
    with tf.name_scope('pixel_wise_softmax'):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def bin_cross_entropy(y_, y):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0), name='bin_cross_entropy'))


class UNet(object):
    def __init__(self):
        self.x = tf.placeholder(
            tf.float32, shape=[None, None, None, IMG_CHANNEL], name='x')
        self.y = tf.placeholder(
            tf.float32, shape=[None, None, None, N_CLASS], name='y')
        self.output_map, self.variables = build_unet(self.x)
        self.cost = self._get_cost(self.output_map)
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

    def _get_cost(self, output_map):
        with tf.name_scope('cost'):
            if HOW_TO_CAL_COST == 'cross_entropy':
                loss = bin_cross_entropy(tf.reshape(self.y, [-1, N_CLASS]),
                                         tf.reshape(pixel_wise_softmax(output_map), [-1, N_CLASS]))
            elif HOW_TO_CAL_COST == 'dice_coefficient':
                pass
            else:
                raise ValueError('未知损失函数计算方法%s' % HOW_TO_CAL_COST)
            regularizer = None
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
