'''
Demo
修改图片的亮度和对比度
以及img和Image之间的转换
'''

import os
import shutil
import cv2
from PIL import Image
import numpy as np
import time

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Array2Image(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def Image2Array(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def modifyBrightnessAndContrast(img, alpha, beta):
    alpha *= 0.01
    img = np.uint8(np.clip((alpha * img + beta), 0, 255))
    cv2.imwrite(savePath + 'alpha_' + str(alpha) + '_' + 'beta_' + str(beta) + ' - ' + pic[pic.find('./') + 2:], img)

    return None

if __name__ == '__main__':
    pic = './4_predicted.jpg'

    # img = cv2.imread(pic)
    # cv_show('img', img)
    # image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # image.show()

    # img = Image.open(pic)
    # img.show()
    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # cv_show('img', img)

    # ###########################################################
    # '''
    # https://blog.csdn.net/weixin_44493841/article/details/102475295?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159964080319724839203176%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=159964080319724839203176&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~pc_rank_v3-1-102475295.pc_ecpm_v3_pc_rank_v3&utm_term=cv2%E8%B0%83%E6%95%B4%E5%9B%BE%E7%89%87%E7%9A%84%E4%BA%AE%E5%BA%A6&spm=1018.2118.3001.4187
    # 亮度调整是将图像像素的强度整体变大/变小，对比度调整指的是图像暗处变得更暗，亮出变得更亮，从而拓宽某个区域内的显示精度。
    # 创建两个滑动条分别调整对比度和亮度（对比度范围：0 ~ 0.3， 亮度0 ~ 100）。提示：因为滑动条没有小数，所以可以设置为0 ~ 300，然后乘以0.01
    # '''
    # alpha = 0.3
    # beta = 80
    # img_path = pic
    # img = cv2.imread(img_path)
    # img2 = cv2.imread(img_path)
    #
    # def updateAlpha(x):
    #     global alpha, img, img2
    #     alpha = cv2.getTrackbarPos('Alpha', 'image')
    #     alpha = alpha * 0.01
    #     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
    #
    #
    # def updateBeta(x):
    #     global beta, img, img2
    #     beta = cv2.getTrackbarPos('Beta', 'image')
    #     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
    #
    # # 创建窗口
    # cv2.namedWindow('image')
    # cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
    # cv2.createTrackbar('Beta', 'image', 0, 255, updateBeta)
    # cv2.setTrackbarPos('Alpha', 'image', 100)
    # cv2.setTrackbarPos('Beta', 'image', 10)
    # while (True):
    #     cv2.imshow('image', img)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    # ###########################################################

    # ############################################################
    # '''
    # https://blog.csdn.net/weixin_43289135/article/details/105315361?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159964080319724839203176%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=159964080319724839203176&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~pc_rank_v3-2-105315361.pc_ecpm_v3_pc_rank_v3&utm_term=cv2%E8%B0%83%E6%95%B4%E5%9B%BE%E7%89%87%E7%9A%84%E4%BA%AE%E5%BA%A6&spm=1018.2118.3001.4187
    # '''
    # # 调整最大值
    # MAX_VALUE = 100
    #
    # def update(input_img_path, output_img_path, lightness, saturation):
    #     """
    #     用于修改图片的亮度和饱和度
    #     :param input_img_path: 图片路径
    #     :param output_img_path: 输出图片路径
    #     :param lightness: 亮度
    #     :param saturation: 饱和度
    #     """
    #
    #     # 加载图片 读取彩色图像归一化且转换为浮点型
    #     image = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    #
    #     # 颜色空间转换 BGR转为HLS
    #     hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #
    #     # 1.调整亮度（线性变换)
    #     hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    #     hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    #     # 饱和度
    #     hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    #     hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    #     # HLS2BGR
    #     lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    #     lsImg = lsImg.astype(np.uint8)
    #     cv2.imwrite(output_img_path, lsImg)
    #
    #
    # dataset_dir = 'imgs'
    # output_dir = 'output'
    #
    # # 这里调参！！！
    # lightness = int(input("lightness(亮度-100~+100):"))  # 亮度
    # saturation = int(input("saturation(饱和度-100~+100):"))  # 饱和度
    #
    # # # 获得需要转化的图片路径并生成目标路径
    # # image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x))
    # #                    for x in os.listdir(dataset_dir)]
    # # # 转化所有图片
    # # for path in image_filenames:
    # #     update(path[0], path[1], lightness, saturation)
    #
    # update(pic, './test.jpg', lightness, saturation)
    # ############################################################

    savePath = './modifiedPics/'
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath)
    img = cv2.imread(pic)
    t0 = time.perf_counter()
    for alpha in range(100, 310, 20):
        for beta in range(0, 110, 20):
            modifyBrightnessAndContrast(img, alpha, beta)
    print(time.perf_counter() - t0)
