import cv2
import numpy as np
from PyQt5.QtGui import *


# cv的数组图像转换成qt的img
def cv_img2qt_img(cv_img):
    cv_img = cv_img.astype(np.uint8)
    height, width, channels = cv_img.shape[:3]
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qt_img = QImage(cv_img.data, width, height, width * channels, QImage.Format_RGB888)
    return qt_img


# 根据比例调整图片大小
def resize_img(img, width, height):
    w = np.array(img).shape[1]
    h = np.array(img).shape[0]

    if w / width >= h / height:
        ratio = w / width
    else:
        ratio = h / height
    new_width = int(w / ratio)
    new_height = int(h / ratio)

    return new_width, new_height
