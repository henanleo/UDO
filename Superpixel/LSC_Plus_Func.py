import imtools
from DoLSCKmeans import DoLSCKmeans
from Initialize import Initialize
from pylab import *
import scipy.io as scio
import numpy as np
from PIL import Image
from scipy.ndimage import correlate


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def my_fspecial(type_str, shape, *args):
    if type_str != "gaussian":
        raise ValueError("type {} not implemented!".format(type_str))
    if isinstance(shape, int):
        shape = [shape, shape]
    if len(args) == 0:
        return matlab_style_gauss2D(shape)
    return matlab_style_gauss2D(shape, args[0])


def my_imread(path: str):
    return np.asarray(Image.open(path), dtype=np.uint8)


def run(img_gray, dsm_seg, name):
    """通过LSC超像素分割的方法对图像进行区域划分"""
    """
    mex wrapper of mexFuncion
    :param I: image data
    :param superpixelNum: number of superpixel to generate
    :param ratio:
    :return: label data
    """

    SHOW_SAVE = 1
    superpixelNum = 20
    ratio = 0.25  # 空间位置(x,y)与颜色（l,a,b）的比值,值越大，越平滑，0.45作为0.4存在
    ratio_dsm = 0.3  # 高程值所占的比例
    img_gray_dsm = img_gray*(1-ratio_dsm) + dsm_seg*ratio_dsm

    rows, cols = img_gray.shape
    img = np.zeros([rows, cols, 3], dtype="uint8")
    img[:, :, 0] = img_gray_dsm
    img[:, :, 1] = img_gray_dsm
    img[:, :, 2] = img_gray_dsm

    dsm = np.zeros([rows, cols, 3], dtype="uint8")
    dsm[:, :, 0] = dsm_seg
    dsm[:, :, 1] = dsm_seg
    dsm[:, :, 2] = dsm_seg

    gaus = my_fspecial('gaussian', 3)

    I = correlate(img.astype(np.float64),
                  gaus.reshape(gaus.shape[0], gaus.shape[1], 1),
                  mode="constant").round().astype(np.uint8)

    dsm_I = correlate(dsm.astype(np.float64),
                      gaus.reshape(gaus.shape[0], gaus.shape[1], 1),
                      mode="constant").round().astype(np.uint8)

    assert len(I.shape) == 3, "The input image must be in CIERGB form"
    assert I.dtype == np.uint8, "The input image must be in CIERGB form"
    nRows, nCols, _ = I.shape
    pixel = nRows * nCols
    label = np.empty([pixel], dtype=np.uint16)

    # 初始化
    (L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, D1, W,
     label, seedArray, newSeedNum, nRows, nCols,
     StepX, StepY, iterationNum, thresholdCoef, new_label) = Initialize(I, dsm_I, nCols, nRows, superpixelNum, ratio,
                                                                        ratio_dsm, label)

    # 执行聚类
    label = DoLSCKmeans(L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, D1, W,
                        label, seedArray, newSeedNum, nRows, nCols,
                        StepX, StepY, iterationNum, thresholdCoef, new_label)

    nRows, nCols, _ = img.shape

    label_2D = label.reshape([nCols, nRows]).transpose([1, 0])

    # 显示并保存分割结果
    if SHOW_SAVE:
        imtools.DisplaySuperpixel(label_2D, img, name)

    return label_2D
