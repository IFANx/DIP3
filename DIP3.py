import cv2 as cv
import random
import numpy as np

# 腐蚀图像
def erode_image(origin_img):
    # origin_img = cv.imread(img_path)

    # gray_img = cv.cvtColor(origin_img, cv.COLOR_BGRGRAY)
    b, g, r = cv.split(origin_img);

    # OpenCV定义的结构元素

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 腐蚀图像

    eroded_b = cv.erode(b, kernel)
    eroded_g = cv.erode(g, kernel)
    eroded_r = cv.erode(r, kernel)

    # 显示腐蚀后的图像
    eroded = cv.merge([eroded_b,eroded_g,eroded_r])

    # cv.imshow('Erode', eroded)
    return eroded



# 膨胀图像
def dilate_image(origin_img):
    # origin_img = cv.imread(img_path)

    # gray_img = cv.cvtColor(origin_img, cv.COLOR_BGRGRAY)
    b,g,r=cv.split(origin_img);
    # OpenCV定义的结构元素

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 膨胀图像

    dilated_b = cv.dilate(b, kernel)
    dilated_g = cv.dilate(g, kernel)
    dilated_r = cv.dilate(r, kernel)

    img_dilated=cv.merge([dilated_b, dilated_g, dilated_r])

    # 显示膨胀后的图像

    # cv.imshow('Dilate', img_dilated)
    return img_dilated



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
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# 读取图片
img = cv.imread("img2.jpg")
# #保存图片
# cv.imwrite("images\\4_original_image.jpg", img)

# 添加椒盐噪声，噪声比例为 0.01
img_pepper = rect_noise(img, prob=0.01)
# 添加椒盐噪声后图像
cv.imshow("add salt_pepper", img_pepper)
# #保存图片
# cv.imwrite("images\\4_pepper_image.jpg", img_pepper)

# 腐蚀操作
# dilate_image1 = dilate_image(img)
erode_image1 = erode_image(img)
# dilate_image2 = dilate_image(dilate_image1)
# dilate_image3 = dilate_image(dilate_image2)
erode_image2 = erode_image(erode_image1)
erode_image3 = erode_image(erode_image2)
erode_image4 = erode_image(erode_image3)

# cv.imshow('Lastdilate', dilate_image3)
cv.imshow('Lasteroded1', erode_image1)
cv.imshow('Lasteroded2', erode_image2)
cv.imshow('Lasteroded3', erode_image3)
cv.imshow('Lasteroded4', erode_image4)

d1= dilate_image(erode_image4)
d2= dilate_image(d1)
d3= dilate_image(d2)
d4= dilate_image(d3)

cv.imshow('res', d4)


# cv.imshow(erode);


cv.waitKey(0)

cv.destroyAllWindows()