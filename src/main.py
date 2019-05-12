from src.train import *

if __name__ == '__main__':
    unet = UNet()
    # i_net = NetDir('../i_net', 'unet')
    images = glob('/Users/guozirui/Desktop/tmp/tyrol/*.tif')
    for image in images:
        unet.predict(image, '/Users/guozirui/Desktop/model/', '.')
