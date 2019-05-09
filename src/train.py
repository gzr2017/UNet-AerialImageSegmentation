from src.unet import *


def update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] *
                            (1.0 - (1.0 / (step + 1)))) + (gradients[i] /
                                                           (step + 1))
    return avg_gradients


def get_a_optimizer(learning_rate, global_step, cost):
    if OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=MOMENTUM).minimize(
            cost,
            global_step=global_step)
    elif OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost,
                                                  global_step=global_step)
    else:
        raise ValueError('未明的优化器%s' % OPTIMIZER)
    return optimizer


def train(unet, i_net, train_dataset):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate_node = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE,
        global_step=global_step,
        decay_steps=train_dataset.iterations,
        decay_rate=DECAY_RATE,
        staircase=True)
    optimizer = get_a_optimizer(learning_rate_node, global_step, unet.cost)
    norm_gradients_node = tf.Variable(tf.constant(
        0.0, shape=[len(unet.gradients_node)]),
        name='norm_gradients')
    tf.summary.scalar('loss', unet.cost)
    tf.summary.scalar('cross_entropy', unet.cross_entropy)
    tf.summary.scalar('accuracy', unet.accuracy)
    if norm_gradients_node is not None:
        tf.summary.histogram('norm_grads', norm_gradients_node)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(i_net.dir_dict['output'], graph=sess.graph)
        if RESTORE is not None:
            unet.restore(sess, i_net.dir_dict['model'])
        logging.info('Start Optimization')
        avg_gradients = None
        for epoch in range(EPOCHS):
            total_loss = 0
            for step in range(epoch * train_dataset.iterations,
                              (epoch + 1) * train_dataset.iterations):
                try:
                    train_data = sess.run(train_dataset.iterator.get_next())
                except tf.errors.OutOfRangeError:
                    logging.error('End of training dataset!')
                train_x = train_data['aerial_image']
                train_y = train_data['aerial_image_label']
                _, summary_str, loss, output_map, gradients, _ = sess.run(
                    (optimizer, summary_op, unet.cost, unet.output_map,
                     unet.gradients_node, learning_rate_node),
                    feed_dict={
                        unet.x: train_x,
                        unet.y: train_y,
                    })
                if NORM_GRADS:
                    avg_gradients = update_avg_gradients(
                        avg_gradients, gradients, step)
                    norm_gradients = [
                        np.linalg.norm(gradient) for gradient in avg_gradients
                    ]
                    norm_gradients_node.assign(norm_gradients).eval()
                total_loss += loss
                unet.save(sess, i_net.dir_dict['model'], next(i_net.net_name_generator))
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                if step % DISPLAY_STEP == 0:
                    logging.info(
                        '迭代到第{}轮；现在的平均loss为：{}'.format(step, total_loss / (step - epoch * train_dataset.iterations)))
                    prediction = output_class(output_map)
                    class_to_color(train_x, train_y, prediction, i_net.dir_dict['prediction'],
                                   str(step))
        logging.info('Optimization Finished!')
