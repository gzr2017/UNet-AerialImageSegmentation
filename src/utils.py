from glob import glob
from PIL import Image
from os import path, makedirs
import numpy as np
from itertools import count, cycle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

color_class_dict = {(0,): 1, (255,): 0}
n_class = len(list(color_class_dict.keys()))


def name_generator(file_path, ex_name=None, cycle_num=None):
    if len(file_path.split('.')) == 2:  # file_path带扩展名
        true_file_name = path.basename(file_path).split('.')[-2]
        true_file_type = path.basename(file_path).split('.')[-1]
    else:  # file_path不带拓展名
        true_file_name = path.basename(file_path)
        true_file_type = None
    if not ex_name:
        true_file_type = None
    elif ex_name is not None:
        true_file_type = ex_name
    if cycle_num is not None:
        for i in cycle(range(cycle_num)):
            if true_file_type is not None:
                yield true_file_name + '_' + str(i) + '.' + true_file_type
            else:
                yield true_file_name + '_' + str(i)
    else:
        for i in count(0):
            if true_file_type is not None:
                yield true_file_name + '_' + str(i) + '.' + true_file_type
            else:
                yield true_file_name + '_' + str(i)


def resize_image(image_path, new_width, new_height, save_dir=None):
    image = Image.open(image_path)
    image = image.resize((new_width, new_height))
    if save_dir is not None:
        file_name = path.basename(image_path)
        image.save(path.join(save_dir, file_name))
    else:
        image.save(image_path)


def split_image(image_path,
                split_width,
                split_height,
                save_dir,
                save_as_img=False):
    # 按下面顺序分割：
    # 1 2 3
    # 4 5 6
    image = Image.open(image_path)
    image_width, image_height = image.size
    if image_width % split_width or image_height % split_height:
        raise ValueError('该图像大小为：{}x{}；不能被等分为{}x{}的图像'.format(
            image_width, image_height, split_width, split_height))
    new_name = name_generator(image_path, ex_name=False)
    for i in range(0, image_height, split_height):
        for j in range(0, image_width, split_width):
            cutting_box = (j, i, j + split_width, i + split_height)
            slice_of_image = image.crop(cutting_box)
            if save_as_img:
                slice_of_image.save(
                    path.join(save_dir,
                              next(new_name) + '.png'))
            else:
                slice_of_image = np.array(slice_of_image)
                np.save(path.join(save_dir, next(new_name)), slice_of_image)


def glut_image(image_list, glut_cols, glut_rows, image_width, image_height,
               save_path):
    glutted_image = Image.new(
        'RGB', (image_width * glut_cols, image_height * glut_rows))
    k = 0
    for i in range(glut_rows):
        for j in range(glut_cols):
            # 判断image_list是不是字符串
            # paste_it = Image.open(image_list[k])
            paste_it = image_list[k]
            glutted_image.paste(paste_it, (j * image_width, i * image_height))
            k += 1
    glutted_image.save(save_path)


def color_to_class(image_path, save_path=None):
    raw_image = np.load(image_path)
    raw_image = np.reshape(raw_image,
                           (raw_image.shape[0], raw_image.shape[1], -1))
    [rows, cols, _] = raw_image.shape
    classed_image = np.zeros((rows, cols, n_class))
    for i in range(rows):
        for j in range(cols):
            classed_image[i, j, color_class_dict[tuple(raw_image[i, j])]] = 1
    if save_path is not None:
        np.save(path.join(save_path,
                          path.basename(image_path).split('.')[-2]),
                classed_image)
    else:
        return classed_image


def output_map_to_class(output_map):
    most_possible_label = np.argmax(output_map, 2)
    classed_image = np.zeros(shape=output_map.shape)
    [rows, cols] = most_possible_label.shape
    for i in range(rows):
        for j in range(cols):
            classed_image[i, j, most_possible_label[i, j]] = 1
    return classed_image


def class_to_color(classed_image):
    reverse_color_class_dict = dict(
        zip(color_class_dict.values(), color_class_dict.keys()))
    colored_image = np.zeros(shape=(classed_image.shape[0],
                                    classed_image.shape[1],
                                    len(list(color_class_dict.keys())[0])))
    for i in range(classed_image.shape[0]):
        for j in range(classed_image.shape[1]):
            for k in range(n_class):
                if classed_image[i][j][k] == 1:
                    colored_image[i][j] = reverse_color_class_dict[k]
    return colored_image
