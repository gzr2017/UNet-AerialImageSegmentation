import numpy as np
from glob import glob
from os import path, makedirs
from src.constant import *
import tensorflow as tf
from PIL import Image
import logging
from itertools import count

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def file_name_generator(file_path=None, ex_name=None, file_name=None):
    # TODO: 受限的循环生成器
    if file_path is not None:
        base_name = path.basename(file_path)
        try:
            true_file_name = base_name.split('.')[-2]
            true_file_type = base_name.split('.')[-1]
        except IndexError:
            logging.error('你的路径中未找到被dot分隔的文件名及拓展名')
        for i in count(0):
            if ex_name is None:  # 如果文件拓展名缺省，那么就用原始的文件拓展名
                yield true_file_name + '_' + str(i) + '.' + true_file_type
            elif not ex_name:  # 如果ex_name==False，不使用文件名
                yield true_file_name + '_' + str(i)
            else:  # 如果文件拓展名不缺省，那么使用该文件拓展名
                yield true_file_name + '_' + str(i) + '.' + ex_name
    elif file_name is not None:
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


def create_dataset(image_data_source, label_data_source, tfrecord_save_path):
    raw_aerial_images = glob(path.join(image_data_source, '*.*'))
    raw_aerial_image_labels = glob(path.join(label_data_source, '*.*'))
    if len(raw_aerial_image_labels) != len(raw_aerial_images):
        raise ValueError('图片数量与标签数量不相等！！！！')
    if len(raw_aerial_images) % DATASET_SLICE:
        raise ValueError('分片数不能整除！！！！！！')
    tfrecord_name_generator = file_name_generator(ex_name='tfrecord',
                                                  file_name='aerial_pair')
    logging.info('>>>开始生成数据集，数据集保存至%s<<<' % tfrecord_save_path)
    for i in range(0, len(raw_aerial_images), DATASET_SLICE):
        aerial_images = raw_aerial_images[i:i + DATASET_SLICE]
        aerial_image_labels = raw_aerial_image_labels[i:i + DATASET_SLICE]
        writer = tf.python_io.TFRecordWriter(
            path.join(tfrecord_save_path, next(tfrecord_name_generator)))
        for aerial_image, aerial_image_label in zip(aerial_images,
                                                    aerial_image_labels):
            aerial_image_raw = np.array(Image.open(aerial_image))
            aerial_image_label_raw = np.load(aerial_image_label)
            feature = {}
            feature['aerial_image'] = tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=aerial_image_raw.flatten()))
            feature['aerial_image_label'] = tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=aerial_image_label_raw.flatten()))
            aerial_pair = tf.train.Example(features=tf.train.Features(
                feature=feature))
            writer.write(aerial_pair.SerializeToString())
        writer.close()


def parse_dataset(proto):
    dics = {
        'aerial_image_label':
        tf.FixedLenFeature(shape=[OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE, N_CLASS],
                           dtype=tf.float32),
        'aerial_image':
        tf.FixedLenFeature(shape=[IMG_SIZE, IMG_SIZE, IMG_CHANNEL],
                           dtype=tf.float32)
    }
    parsed_pair = tf.parse_single_example(proto, dics)
    return parsed_pair


def get_data_iterator(tfrecord_path):
    filenames = glob(path.join(tfrecord_path, '*.tfrecord'))
    if len(filenames) == 0:
        raise ValueError('指定目录下未找到tfrecord文件！！！')
    logging.info('>>>从%s取得数据集<<<' % tfrecord_path)
    dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = dataset.map(parse_dataset)
    parsed_dataset = parsed_dataset.repeat(EPOCHS)
    parsed_dataset = parsed_dataset.shuffle(buffer_size=10000)
    parsed_dataset = parsed_dataset.batch(BATCH_SIZE)
    iterator = parsed_dataset.make_one_shot_iterator()
    return iterator


class DatasetDir(object):
    def __init__(self, base_dir):
        self.dir_dict = {
            'original_data': path.join(base_dir, 'original/data'),  # 原始data
            'original_label': path.join(base_dir, 'original/label'),  # 原始label
            'split_data': path.join(base_dir, 'split/data'),  # 分割后data
            'split_label': path.join(base_dir, 'split/label'),  # 分割后label
            # 将color转为class之后的label
            'split_label_classed': path.join(base_dir, 'split/label_classed'),
            'tfrecord': path.join(base_dir, 'tfrecord'),  # tfrecord存储位置
        }
        if len(glob(path.join(self.dir_dict['original_data'], '*.*'))) == 0:
            raise ValueError('文件夹%s中未找到任何文件！')
        if len(glob(path.join(self.dir_dict['original_label'], '*.*'))) == 0:
            raise ValueError('文件夹%s中未找到任何文件！')
        for dir in self.dir_dict.values():
            if not path.exists(dir):
                makedirs(dir)

    def data_nagare(self,
                    is_split=True,
                    is_tran_class=True,
                    is_create_dataset=True):
        if is_split and len(glob(path.join(self.dir_dict['split_data'],
                                           '*.*'))) == 0:
            split_img(self.dir_dict['original_data'],
                      self.dir_dict['split_data'])
        if is_split and len(
                glob(path.join(self.dir_dict['split_label'], '*.*'))) == 0:
            split_img(self.dir_dict['original_label'],
                      self.dir_dict['split_label'], OUTPUT_IMG_SIZE)
        if is_tran_class and len(
                glob(path.join(self.dir_dict['split_label_classed'],
                               '*.*'))) == 0:
            color_to_class(self.dir_dict['split_label'],
                           self.dir_dict['split_label_classed'])
        if is_create_dataset and len(
                glob(path.join(self.dir_dict['tfrecord'], '*.*'))) == 0:
            create_dataset(self.dir_dict['split_data'],
                           self.dir_dict['split_label_classed'],
                           self.dir_dict['tfrecord'])

    def get_a_iterator(self):
        label_num = len(
            glob(path.join(self.dir_dict['split_label_classed'], '*.*')))
        data_num = len(glob(path.join(self.dir_dict['split_data'], '*.*')))
        if data_num == 0 or label_num == 0:
            logging.info('你似乎未完成文件分割工作惹')
        self.iterations = data_num * EPOCHS // BATCH_SIZE
        self.iterator = get_data_iterator(self.dir_dict['tfrecord'])


class NetDir(object):
    def __init__(self, base_dir, net_name):
        self.dir_dict = {
            'model': path.join(base_dir, 'model'),
            'output': path.join(base_dir, 'output'),
            'prediction': path.join(base_dir, 'prediction'),
        }
        for dir in self.dir_dict.values():
            if not path.exists(dir):
                makedirs(dir)
        self.net_name_generator = file_name_generator(file_name=net_name,
                                                      ex_name='ckpt')
