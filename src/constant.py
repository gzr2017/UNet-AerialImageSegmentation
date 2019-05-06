# ref:
# https://github.com/jakeret/tf_unet/blob/master/tf_unet/unet.py
# https://www.jianshu.com/p/6cee728f3490
# https://blog.csdn.net/whitesilence/article/details/75041247
# https://www.chinaopen.ai/industryDynamic/competition-26.html
# https://www.zhihu.com/question/49346370

RESTORE = None  # 是否重新读取
SUMMARIES = True
# 输入像的大小和通道数
IMG_SIZE = 250  # 图像尺寸不能设置得太小了，要不然池化池化池化得没有了……
IMG_CHANNEL = 3
OUTPUT_IMG_SIZE = 52  # 手工计算……我也想不到什么可以自动计算的方法惹
N_CLASS = 2
COLOR_CLASS_DICT = {  # 注意！！！！！！！！！OpenCV是按照BGR顺序存储颜色的！！！！！！！！！！虽然这里并没有使用OpenCV……如果用了OpenCV还是注意一个比较好
    (0,): 0,  # 背景：即使是单通道图片，也要打逗号！！！！！！！！！！
    (255,): 1  # 房屋
}
# 训练参数
BATCH_SIZE = 4
EPOCHS = 10
OPTIMIZER = 'momentum'  # 选择
HOW_TO_CAL_COST = 'cross_entropy'  # 选择损失函数计算方法
MOMENTUM = 0.2
LEARNING_RATE = 1e-4
DECAY_RATE = 0.95
DISPLAY_STEP = 5  # number of steps till outputting stats
NORM_GRADS = True
DATASET_SLICE = 10  # 每个tfrecord中包含多少对数据
# UNet网络参数
LAYERS = 4
FEATURES_ROOT = 64
FILTER_SIZE = 3
POOL_SIZE = 2
PADDING_WAY = 'VALID'

# RESTORE = None  # 是否重新读取
# SUMMARIES = True
# # 输入像的大小和通道数
# IMG_SIZE = 256  # 图像尺寸不能设置得太小了，要不然池化池化池化得没有了……
# IMG_CHANNEL = 1
# OUTPUT_IMG_SIZE = 256  # 手工计算……我也想不到什么可以自动计算的方法惹
# N_CLASS = 2
# COLOR_CLASS_DICT = {  # 注意！！！！！！！！！OpenCV是按照BGR顺序存储颜色的！！！！！！！！！！虽然这里并没有使用OpenCV……如果用了OpenCV还是注意一个比较好
#     (0,): 1,  # 背景：即使是单通道图片，也要打逗号！！！！！！！！！！
#     (255,): 0  # 房屋
# }
# # 训练参数
# BATCH_SIZE = 4
# EPOCHS = 20
# OPTIMIZER = 'adam'  # 选择
# HOW_TO_CAL_COST = 'cross_entropy'  # 选择损失函数计算方法
# MOMENTUM = 0.2
# LEARNING_RATE = 1e-5
# DECAY_RATE = 0.95
# DISPLAY_STEP = 5  # number of steps till outputting stats
# NORM_GRADS = True
# DATASET_SLICE = 10  # 每个tfrecord中包含多少对数据
# # UNet网络参数
# LAYERS = 4
# FEATURES_ROOT = 64
# FILTER_SIZE = 3
# POOL_SIZE = 2
# PADDING_WAY = 'SAME'
