import numpy as np
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import DSMtools
import scipy.io as scio


def uint16_to_uint8(img):
    """本函数实现：uint16 到 uint8的转化"""
    tmp = img / 65535
    img_8 = np.uint8(tmp*255)
    return img_8


def imresize(im, sz):
    """ 使用 PIL 对象重新定义图像数组的大小
    sz[0]:列数--宽
    sz[1]:行数--高
    """
    pil_im = Image.fromarray(uint8(im))
    return np.array(pil_im.resize(sz))


def histeq(im, nbr_bins=65536):
    """ 对一幅灰度图像进行直方图均衡化 """
    # 计算图像的直方图
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 65535 * cdf / cdf[-1]  # 归一化
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


def do_preprocessor(dsm_ori):
    """本函数实现对图像的预处理"""

    # 从uint16转化为uint8
    dsm_ori = uint16_to_uint8(dsm_ori)

    # 直方图均衡化
    dsm, cdf = histeq(dsm_ori)

    # 显示图像
    plt.gray()
    plt.imshow(dsm)
    plt.show()

    return dsm


def do_remove_nan(input_data):
    """remove nan in input_data"""

    # 确定位置
    ind = ~np.isnan(input_data[3, :])
    data_without_nan = input_data[:, ind]

    # print(input_data)
    return data_without_nan


def do_remove_outliers(input_data, min_h=10, max_h=50):
    """remove otliers in input_data"""
    data_new = input_data[input_data > min_h]
    data_new1 = data_new[data_new < max_h]

    return data_new1


def do_CheckRegistration(img, dsm_seg):
    """本函数实现：配准检查"""
    rows, cols = dsm_seg.shape
    dsm_seg_str = DSMtools.dsm_Stretch2Uint8(dsm_seg)

    img_new = np.zeros((rows, cols, 3), dtype="uint8")
    img_new[:, :, 0] = img
    img_new[:, :, 1] = dsm_seg_str
    # img_new[:, :, 2] = dsm_08_str

    # 显示并输出配准结果
    img_new = Image.fromarray(img_new)  # 转换格式再显示
    # img_new.show()
    # img_new.save("R01_results_prop/R1_img_DSM_registration.png")  # 保存R1图像
    img_new.save("R02A_results_prop/img_R02A_DSM_registration.png")  # 保存R2图像


def DisplaySuperpixel(label_2D: np.ndarray, img: np.ndarray, name):

    img = img.copy()
    nRows, nCols = label_2D.shape
    for i in range(nRows):
        for j in range(nCols):
            minX = 0 if i - 1 < 0 else i - 1
            minY = 0 if j - 1 < 0 else j - 1
            maxX = nRows - 1 if i + 1 >= nRows else i + 1
            maxY = nCols - 1 if j + 1 >= nCols else j + 1
            count = (label_2D[minX:maxX + 1, minY:maxY + 1] != label_2D[i][j]).sum()
            if count >= 2:
                img[i][j] = [255, 0, 0]
    PIL_image = Image.fromarray(img, 'RGB')
    # PIL_image.show()

    name_canshu = "_LSCplus"
    PIL_image.save("result_R01/" + name.split(".")[0] + name_canshu + ".png")

    LSCplus_dir = "result_R01/" + name.split(".")[0] + name_canshu + ".mat"
    scio.savemat(LSCplus_dir, {'label_2D': label_2D})  # 保存mat文件
    # print('OK')


def DisplayLabel(label_2D: np.ndarray, name, resultPath):
    nRows, nCols = label_2D.shape
    img = np.zeros([nRows, nCols, 3], dtype=np.uint8)
    for i in range(nRows):
        for j in range(nCols):
            minX = 0 if i - 1 < 0 else i - 1
            minY = 0 if j - 1 < 0 else j - 1
            maxX = nRows - 1 if i + 1 >= nRows else i + 1
            maxY = nCols - 1 if j + 1 >= nCols else j + 1
            count = (label_2D[minX:maxX + 1, minY:maxY + 1] != label_2D[i][j]).sum()
            if count >= 2:
                img[i][j] = [255, 255, 255]
    PIL_image = Image.fromarray(img, 'RGB')
    PIL_image.show()
    PIL_image.save(resultPath + name.split(".")[0] + "_LSC" + ".jpg")
