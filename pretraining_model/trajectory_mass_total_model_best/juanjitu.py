import cv2
import numpy as np
import matplotlib.pyplot as plt


def myConv(x, kernel, padding):
    H, W = x.shape
    h, w = kernel.shape
    n = h // 2
    if padding == 'zero':
        x = cv2.copyMakeBorder(x, n, n, n, n, cv2.BORDER_CONSTANT, value=0)
    elif padding == 'reflect':
        x = cv2.copyMakeBorder(x, n, n, n, n, cv2.BORDER_REFLECT)
    elif padding == 'replicate':
        x = cv2.copyMakeBorder(x, n, n, n, n, cv2.BORDER_REPLICATE)

    img = np.zeros((H, W), dtype=np.float)
    for i in range(H):
        for j in range(W):
            img[i, j] = (x[i:i + h, j:j + w] * kernel).sum()

    img = img.astype(np.uint8)
    return img


# Gaussian
des = r'.\dog.jpg'
im = cv2.imread(des, 0)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
Gauss = cv2.GaussianBlur(im, (5, 5), 1.3)
Kernel = cv2.getGaussianKernel(3, 1.3)
kernel = np.multiply(Kernel, np.transpose(Kernel.copy()))
Gaussmy = myConv(im, kernel, 'reflect')
Mask = im-Gauss
Mask1 = Mask.copy().astype(np.uint8)
# plt.subplots_adjust(left=0.15, bottom=0.1, top=1, right=1, hspace=0.5, wspace=0.5)
plt.subplot(221)
plt.imshow(im, cmap='gray')
plt.title('原狗图')
plt.subplot(222)
plt.imshow(Gauss, cmap='gray')
plt.title('内置函数实现的狗')
plt.subplot(223)
plt.imshow(Gaussmy, cmap='gray')
plt.title('自定义函数实现的狗')
plt.subplot(224)
plt.imshow(Mask1, cmap='gray')
plt.title('狗的掩膜')
plt.show()