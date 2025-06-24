import math
import cmath
from Seeds import gen_seeds
from skimage import color
import time
import numpy as np

PI = 3.1415926


def myrgb2lab(I: np.ndarray, row_num: int, col_num: int):
    """
    change rgb to lab format
    :param I: rgb format image
    :return:
        L: L channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        a: a channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        b: b channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
    """
    lab_img = color.rgb2lab(I).transpose([2, 1, 0])
    L = lab_img[0].copy().reshape([row_num * col_num])
    a = lab_img[1].copy().reshape([row_num * col_num])
    b = lab_img[2].copy().reshape([row_num * col_num])
    L /= (100 / 255)  # L is [0, 100], change it to [0, 255]
    L += 0.5
    a += 128 + 0.5  # A is [-128, 127], change it to [0, 255]
    b += 128 + 0.5  # B is [-128, 127], change it to [0, 255]
    return L.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)


def Initialize_Func(L: np.ndarray, a: np.ndarray, b: np.ndarray, Dsm: np.ndarray, nRows: int, nCols: int, StepX: int, StepY: int,
                    Color: float, Distance: float, dsmDist: float):
    print("\t[{}] [Initialize.py] step_1/3".format(time.ctime()[11:19]))
    vcos = np.vectorize(math.cos)
    vsin = np.vectorize(math.sin)
    thetaL = (np.resize(L.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetaa = (np.resize(a.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetab = (np.resize(b.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetax = np.empty([nRows, nCols], dtype=np.float64)
    thetay = np.empty([nRows, nCols], dtype=np.float64)
    for i in range(thetax.shape[0]):
        thetax[i, :] = i
    for j in range(thetay.shape[1]):
        thetay[:, j] = j
    thetax = (thetax / StepX) * PI / 2
    thetay = (thetay / StepY) * PI / 2
    L1 = Color * vcos(thetaL)
    L2 = Color * vsin(thetaL)
    a1 = Color * vcos(thetaa) * 2.55
    a2 = Color * vsin(thetaa) * 2.55
    b1 = Color * vcos(thetab) * 2.55
    b2 = Color * vsin(thetab) * 2.55
    x1 = Distance * vcos(thetax)
    x2 = Distance * vsin(thetax)
    y1 = Distance * vcos(thetay)
    y2 = Distance * vsin(thetay)

    # 高程有点区别是，高程采用的是归一化（除以最大值）
    theta_dsm = (np.resize(Dsm.copy(), [nRows, nCols]) / np.max(Dsm)).astype(np.float64)
    D1 = dsmDist * theta_dsm  # 高程
    print("\t[{}] [Initialize.py] step_2/3".format(time.ctime()[11:19]))
    size = nRows * nCols
    sigmaL1 = L1.sum() / size
    sigmaL2 = L2.sum() / size
    sigmaa1 = a1.sum() / size
    sigmaa2 = a2.sum() / size
    sigmab1 = b1.sum() / size
    sigmab2 = b2.sum() / size
    sigmax1 = x1.sum() / size
    sigmax2 = x2.sum() / size
    sigmay1 = y1.sum() / size
    sigmay2 = y2.sum() / size
    sigmaD1 = D1.sum() / size

    print("\t[{}] [Initialize.py] step_3/3".format(time.ctime()[11:19]))
    W = L1 * sigmaL1 + L2 * sigmaL2 + a1 * sigmaa1 + a2 * sigmaa2 + b1 * sigmab1 + \
        b2 * sigmab2 + x1 * sigmax1 + x2 * sigmax2 + y1 * sigmay1 + y2 * sigmay2 + D1 * sigmaD1
    L1 /= W
    L2 /= W
    a1 /= W
    a2 /= W
    b1 /= W
    b2 /= W
    x1 /= W
    x2 /= W
    y1 /= W
    y2 /= W
    D1 /= W
    return L1.astype(np.float32), L2.astype(np.float32), a1.astype(np.float32), \
           a2.astype(np.float32), b1.astype(np.float32), b2.astype(np.float32), \
           x1.astype(np.float32), x2.astype(np.float32), y1.astype(np.float32), \
           y2.astype(np.float32), D1.astype(np.float32), W.astype(np.float64)


def Initialize(I: np.ndarray, dsm_I: np.ndarray, nRows: int, nCols: int, superpixelnum: int, ratio: float, ratio_dsm: float, label: np.ndarray):
    """开始执行superpixel segmentation algorithm
    """
    new_label = np.empty([nRows, nCols], dtype=np.uint16)
    print("[{}] Setting Parameter...".format(time.ctime()[11:19]))
    colorCoefficient = 20
    distCoefficient = colorCoefficient * ratio
    dsmCoefficient = colorCoefficient * ratio_dsm  # 新增
    seedNum = superpixelnum
    iterationNum = 15
    thresholdCoef = 4

    print("[{}] Translating image from RGB format to LAB format...".format(time.ctime()[11:19]))
    L, a, b = myrgb2lab(I, nRows, nCols)

    # elevation
    Dsm = dsm_I[:, :, 0]
    Dsm = np.array(Dsm).flatten()

    print("[{}] Producing Seeds...".format(time.ctime()[11:19]))
    ColNum = int(cmath.sqrt(seedNum * nCols / nRows).real)
    RowNum = int(seedNum / ColNum)
    StepX = int(nRows / RowNum)
    StepY = int(nCols / ColNum)
    seedArray = gen_seeds(row_num=nRows, col_num=nCols, seed_num=seedNum)
    newSeedNum = len(seedArray)

    print("[{}] Initialization...".format(time.ctime()[11:19]))
    L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, D1, W = Initialize_Func(L, a, b, Dsm, nRows, nCols, StepX, StepY, colorCoefficient,
                                                                distCoefficient, dsmCoefficient)

    # del L
    # del a
    # del b

    print("[{}] Producing Superpixel...".format(time.ctime()[11:19]))
    return L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, D1, W, label, seedArray, newSeedNum, nRows, nCols, StepX, StepY, iterationNum, thresholdCoef, new_label
