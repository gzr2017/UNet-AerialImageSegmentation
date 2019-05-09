import numpy as np
from glob import glob
from os import path, makedirs
from src.constant import *
import tensorflow as tf
from PIL import Image
import logging
from itertools import count, cycle
import cv2


def file_name_generator(file_path=None, ex_name=None, file_name=None, cycle_num=None):
    if file_path is not None:
        base_name = path.basename(file_path)
        try:
            true_file_name = base_name.split('.')[-2]
            true_file_type = base_name.split('.')[-1]
        except IndexError:
            logging.error('你的路径中未找到被dot分隔的文件名及拓展名')
        if cycle_num is not None:
            for i in cycle(range(cycle_num)):
                if ex_name is None:  # 如果文件拓展名为None，那么就用原始的文件拓展名
                    yield true_file_name + '_' + str(i) + '.' + true_file_type
                elif not ex_name:  # 如果ex_name==False，不使用文件名
                    yield true_file_name + '_' + str(i)
                else:  # 如果文件拓展名不缺省，那么使用该文件拓展名
                    yield true_file_name + '_' + str(i) + '.' + ex_name
        else:
            for i in count(0):
                if ex_name is None:  # 如果文件拓展名为None，那么就用原始的文件拓展名
                    yield true_file_name + '_' + str(i) + '.' + true_file_type
                elif not ex_name:  # 如果ex_name==False，不使用文件名
                    yield true_file_name + '_' + str(i)
                else:  # 如果文件拓展名不缺省，那么使用该文件拓展名
                    yield true_file_name + '_' + str(i) + '.' + ex_name
    elif file_name is not None:
        if cycle_num is not None:
            for i in cycle(range(cycle_num)):
                if ex_name is None:
                    yield file_name + '_' + str(i)
                else:
                    yield file_name + '_' + str(i) + '.' + ex_name
        else:
            for i in count(0):
                if ex_name is None:
                    yield file_name + '_' + str(i)
                else:
                    yield file_name + '_' + str(i) + '.' + ex_name
    else:
        raise ValueError('你既没指定目录也没指定文件名:(')


def batch_resize(img_path, resize, save_path=None):
    image_paths = glob(path.join(img_path, '*.*'))
    if len(image_paths) == 0:
        raise ValueError('目录%s下未找到图片' % img_path)
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.resize((resize, resize))
        if save_path is not None:
            file_name = path.basename(image_path)
            image.save(path.join(save_path, file_name))
        else:
            image.save(image_path)


def split_img(img_path, save_path, size=None):
    # TODO: 增加对非2的次方倍的切割工具
    image_paths = glob(path.join(img_path, '*.*'))
    if len(image_paths) == 0:
        raise ValueError('目录%s下未找到图片' % img_path)
    logging.info('>>>开始将目录%s下的图片分割<<<' % img_path)
    for image_path in image_paths:
        image = Image.open(image_path)
        image_width, image_height = image.size
        if image_width % IMG_SIZE or image_height % IMG_SIZE:
            raise ValueError('该图片不能等分为大小%i的图片！！' % IMG_SIZE)
        new_name = file_name_generator(file_path=image_path)
        for i in range(0, image_height, IMG_SIZE):
            for j in range(0, image_width, IMG_SIZE):
                cutting_box = (j, i, j + IMG_SIZE, i + IMG_SIZE)
                slice_of_image = image.crop(cutting_box)
                if size is not None:
                    slice_of_image = slice_of_image.resize((size, size))
                slice_of_image.save(path.join(save_path, next(new_name)))


def color_to_class(img_path, save_path):
    aerial_image_labels = glob(path.join(img_path, '*.*'))
    if len(aerial_image_labels) == 0:
        raise ValueError('目录%s下没有图片！' % img_path)
    logging.info('>>>开始将目录%s下的图片转换为数组<<<' % img_path)
    for aerial_image_label in aerial_image_labels:
        raw_aerial_image_label = np.array(Image.open(aerial_image_label))
        raw_aerial_image_label = np.reshape(raw_aerial_image_label, [
            raw_aerial_image_label.shape[0], raw_aerial_image_label.shape[1],
            -1
        ])
        [rows, cols, _] = raw_aerial_image_label.shape
        label_value = np.zeros((rows, cols, N_CLASS))
        for i in range(rows):
            for j in range(cols):
                label_value[i, j, COLOR_CLASS_DICT[tuple(
                    raw_aerial_image_label[i, j])]] = 1
        np.save(
            path.join(save_path,
                      path.basename(aerial_image_label).split('.')[-2]),
            label_value)


def output_class(output_map):
    '''
    将输出图变成label；
    :param output_map: 输出图
    :return: prediction: 标签化的图像
    '''
    if len(output_map.shape) != 4:
        raise ValueError('输出图不是4维的图像！输出图形状应为[batch_size, rows, cols, channels]')
    most_possible_label = np.argmax(output_map, 3)
    prediction = np.zeros(shape=output_map.shape)
    [batch_size, rows, cols] = most_possible_label.shape
    for h in range(batch_size):
        for i in range(rows):
            for j in range(cols):
                prediction[h, i, j,
                           most_possible_label[h, i, j]] = 1
    return prediction


def class_to_color(data, label, prediction, save_path, save_name):
    # TODO: 这里行列好像有点问题
    reverse_COLOR_CLASS_DICT = dict(
        zip(COLOR_CLASS_DICT.values(), COLOR_CLASS_DICT.keys()))
    [batch_size, rows, cols, _] = prediction.shape
    for h in range(batch_size):
        colored_prediction = np.zeros(
            (rows, cols, len(list(COLOR_CLASS_DICT.keys())[0])),
            dtype=np.uint8)
        recovered_label = np.zeros(
            (rows, cols, len(list(COLOR_CLASS_DICT.keys())[0])),
            dtype=np.uint8)
        single_data = data[h]
        for i in range(rows):
            for j in range(cols):
                for k in range(N_CLASS):
                    if prediction[h][i][j][k] == 1:
                        colored_prediction[i][j] = reverse_COLOR_CLASS_DICT[k]
                    if label[h][i][j][k] == 1:
                        recovered_label[i][j] = reverse_COLOR_CLASS_DICT[k]
        if single_data.shape[2] == 1:
            single_data = np.reshape(single_data, [rows, cols])
        if len(list(COLOR_CLASS_DICT.keys())[0]) == 1:
            recovered_label = np.reshape(recovered_label, [rows, cols])
            colored_prediction = np.reshape(colored_prediction, [rows, cols])
        # 因为池化的舍入，输入图的大小和输出图的大小可能不一样，这里就无脑resize了
        single_data = Image.fromarray(single_data.astype('uint8')).resize((rows, cols))
        recovered_label = Image.fromarray(recovered_label)
        colored_prediction = Image.fromarray(colored_prediction)
        compare_result = Image.new('RGB', (cols * 3, rows))
        compare_result.paste(single_data, (0, 0))
        compare_result.paste(recovered_label, (cols, 0))
        compare_result.paste(colored_prediction, (rows * 2, 0))
        compare_result.save(path.join(save_path, save_name + '_' + str(h) + '.png'))
