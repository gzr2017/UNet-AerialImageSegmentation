from glob import glob
from PIL import Image


def glut_together():
    images = sorted(glob('*.png'))
    glut_t = Image.new('RGB', (4096, 4096))
    k = 0
    for i in range(4):
        for j in range(4):
            fuckimage = Image.open(images[k])
            print(images[k])
            glut_t.paste(fuckimage, (j * 1024, i * 1024))
            k += 1
    glut_t.save('glut.png')


if __name__ == '__main__':
    glut_together()