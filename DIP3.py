import cv2 as cv
import random
import numpy as np


def erode_image(img_path):
    origin_img = cv.imread(img_path)

    gray_img = cv.cvtColor(origin_img, cv.COLOR_BGRGRAY)

    # OpenCV定义的结构元素

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 腐蚀图像

    eroded = cv.erode(gray_img, kernel)

    # 显示腐蚀后的图像

    cv.imshow('Origin', origin_img)

    cv.imshow('Erode', eroded)



def dilate_image(img_path):
    origin_img = cv.imread(img_path)

    gray_img = cv.cvtColor(origin_img, cv.COLOR_BGRGRAY)

    # OpenCV定义的结构元素

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 膨胀图像

    dilated = cv.dilate(gray_img, kernel)

    # 显示腐蚀后的图像

    cv.imshow('Dilate', dilated)



# 添加矩形噪声function(), prob:噪声比例
def rect_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 55
            else:
                output[i][j] = image[i][j]
    return output

# 读取图片
img = cv.imread("car.jpg")
# #保存图片
# cv.imwrite("images\\4_original_image.jpg", img)

# 添加椒盐噪声，噪声比例为 0.01
img_pepper = rect_noise(img, prob=0.01)
# 添加椒盐噪声后图像
cv.imshow("add salt_pepper", img_pepper)
# #保存图片
# cv.imwrite("images\\4_pepper_image.jpg", img_pepper)


cv.waitKey(0)

cv.destroyAllWindows()