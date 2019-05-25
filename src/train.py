from src.cnn import *
import matplotlib.pyplot as plt


def train(
        cnn,  # 网络
        net_dir,  # 网络保存位置
        dataset_iterator,  # 数据集
        iterations,
        learning_rate=1e-5,  # 学习率
        # decay_rate=0.95,
        epochs=1,
        restore=None,
        display_step=10):
    global_step = tf.Variable(0, trainable=False)
    # learning_rate_node = tf.train.exponential_decay(
    #     learning_rate=learning_rate,
    #     global_step=global_step,
    #     decay_steps=iterations,
    #     decay_rate=decay_rate,
    #     staircase=False)
    net_name_generator = name_generator('cnn', 'ckpt', display_step)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cnn.cost,
                                              global_step=global_step)
    tf.summary.scalar('cost', cnn.cost)
    tf.summary.scalar('cross_entropy', cnn.cross_entropy)
    tf.summary.scalar('accuracy', cnn.accuracy)
    tf.summary.scalar('f1_score', cnn.f1_score)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(net_dir['log'],
                                               graph=sess.graph)
        if restore is not None:
            cnn.restore(sess, restore)
        logging.info('>>>Start Optimization!<<<')
        for epoch in range(epochs):
            for step in range(epoch * iterations, (epoch + 1) * iterations):
                try:
                    train_dataset = sess.run(dataset_iterator.get_next())
                except tf.errors.OutOfRangeError:
                    logging.error('End of training dataset!')
                x = train_dataset['data']
                y = train_dataset['label']
                _, cost, accuracy, f1_score, cross_entropy, summary_str, output_map = sess.run(
                    (optimizer, cnn.cost, cnn.accuracy, cnn.f1_score, cnn.cross_entropy,
                     summary_op, cnn.output_map),
                    feed_dict={
                        cnn.x: x,
                        cnn.y: y
                    })
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                if step % display_step == 0:
                    logging.info('迭代到第{}轮；现在的cost为：{}；accuracy为：{}；F1 Score为：{}；交叉熵为：{}'.format(
                        step, cost, accuracy, f1_score, cross_entropy))
                    cnn.save(
                        sess,
                        path.join(net_dir['model'], next(net_name_generator)))
                    # save_train_result(x, y, output_map, net_dir['prediction'], step)


def save_train_result(x, y, output_map, save_dir, step):
    batch_size, rows, cols, _ = x.shape
    for h in range(batch_size):
        recovered_data = Image.fromarray(np.uint8(x[h]))
        recovered_label = class_to_color(y[h])
        colored_prediction = class_to_color(output_map_to_class(output_map[h]))
        if recovered_label.shape[-1] == 1:
            recovered_label = recovered_label.reshape(
                (recovered_label.shape[0], recovered_label.shape[1]))
        if colored_prediction.shape[-1] == 1:
            colored_prediction = colored_prediction.reshape(
                (colored_prediction.shape[0], colored_prediction.shape[1]))
        recovered_label = Image.fromarray(np.uint8(recovered_label))
        colored_prediction = Image.fromarray(np.uint8(colored_prediction))
        img_list = [recovered_label, colored_prediction, recovered_data]
        glut_image(img_list, 3, 1, cols, rows,
                   path.join(save_dir,
                             str(step) + '_' + str(h) + '.png'))
