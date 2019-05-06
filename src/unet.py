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
            (rows, cols, len(list(COLOR_CLASS_DICT.keys())[0])),
            dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                for k in range(N_CLASS):
                    if classed_prediction[h][i][j][k] == 1:
                        color_value[i][j][0:1] = reverse_COLOR_CLASS_DICT[k]
                        break
        if color_value.shape[2] == 1:
            color_value = np.reshape(color_value, [rows, cols])
        real_out_predict_image = Image.fromarray(color_value.astype('uint8'))
        real_out_predict_image.save(
            path.join(save_path, save_name + '_' + str(h) + '.tif'))


def pixel_wise_softmax(output_map):
    with tf.name_scope('pixel_wise_softmax'):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def bin_cross_entropy(y_, y, pos_weight=True):
    y_ = tf.reshape(y_, [-1, N_CLASS])
    y = tf.reshape(pixel_wise_softmax(y), [-1, N_CLASS])
    if not pos_weight:
        return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0),
                                           name='bin_cross_entropy'))
    else:
        count_neg = tf.reduce_sum(1. - y_)
        count_pos = tf.reduce_sum(y_)
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=y,
                                                        targets=y_,
                                                        pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
        return tf.where(zero, 0.0, cost, name='bin_cross_entropy')


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
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self):
        with tf.name_scope('cost'):
            if HOW_TO_CAL_COST == 'bin_cross_entropy':
                logging.warning('使用二分类cross_entropy只可用于计算二分类问题！！！！')
                loss = bin_cross_entropy(self.y, self.output_map, True)
            elif HOW_TO_CAL_COST == 'cross_entropy':
                pass
                #TODO: 补充多分类问题的cross_entropy函数
            elif HOW_TO_CAL_COST == 'dice_coefficient':
                pass
                #TODO: 补充多分类问题的cross_entropy函数
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
        test_x = np.reshape(test_x,
                            newshape=[1, test_x.shape[0], test_x.shape[1], -1])
        with tf.Session() as sess:
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
        saver = tf.train.Saver()
        saver.restore(sess, path.join(save_path, RESTORE))