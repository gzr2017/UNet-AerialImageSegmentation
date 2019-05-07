import matplotlib.pyplot as plt
from PIL import Image

data = Image.open('/Users/guozirui/Downloads/chi1.png')
label = Image.open('/Users/guozirui/Downloads/chi2.png')
plt.figure()
plt.subplot(1, 2, 1)
plt.title('data')
plt.imshow(data), plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('label')
plt.imshow(label), plt.axis('off')
plt.show()
