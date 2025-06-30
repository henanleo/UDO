import numpy as np
import sys
import time
from EnforceConnectivity import EnforceConnectivity
from preEnforceConnetivity import preEnforceConnectivity

DBL_MAX = sys.float_info[0]  # max float value


def DoLSCKmeans(L1: np.ndarray, L2: np.ndarray, a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, D1: np.ndarray,
                W: np.ndarray, label: np.ndarray, seedArray: list, seedNum: int, nRows: int, nCols: int, StepX: int,
                StepY: int, iterationNum: int, thresholdCoef: int, new_label: np.ndarray):

    print("\t[{}] [DoSuperpixel.py]: Pre-treatment".format(time.ctime()[11:19]))
    dist = np.empty([nRows, nCols], dtype=np.float64)
    centerL1 = np.empty([seedNum], dtype=np.float64)
    centerL2 = np.empty([seedNum], dtype=np.float64)
    centera1 = np.empty([seedNum], dtype=np.float64)
    centera2 = np.empty([seedNum], dtype=np.float64)
    centerb1 = np.empty([seedNum], dtype=np.float64)
    centerb2 = np.empty([seedNum], dtype=np.float64)
    centerx1 = np.empty([seedNum], dtype=np.float64)
    centerx2 = np.empty([seedNum], dtype=np.float64)
    centery1 = np.empty([seedNum], dtype=np.float64)
    centery2 = np.empty([seedNum], dtype=np.float64)
    centerD1 = np.empty([seedNum], dtype=np.float64)  # 高程
    WSum = np.empty([seedNum], dtype=np.float64)
    clusterSize = np.empty([seedNum], dtype=np.int32)

    print("\t[{}] [DoSuperpixel.py]: Initialization".format(time.ctime()[11:19]))
    for i in range(seedNum):
        centerL1[i] = 0
        centerL2[i] = 0
        centera1[i] = 0
        centera2[i] = 0
        centerb1[i] = 0
        centerb2[i] = 0
        centerx1[i] = 0
        centerx2[i] = 0
        centery1[i] = 0
        centery2[i] = 0
        centerD1[i] = 0  # 高程
        x = seedArray[i].x
        y = seedArray[i].y
        minX = int(0 if x - StepX // 4 <= 0 else x - StepX // 4)
        minY = int(0 if y - StepY // 4 <= 0 else y - StepY // 4)
        maxX = int(nRows - 1 if x + StepX // 4 >= nRows - 1 else x + StepX // 4)
        maxY = int(nCols - 1 if y + StepY // 4 >= nCols - 1 else y + StepY // 4)
        Count = 0
        for j in range(minX, maxX + 1):
            for k in range(minY, maxY + 1):
                Count += 1
                centerL1[i] += L1[j][k]
                centerL2[i] += L2[j][k]
                centera1[i] += a1[j][k]
                centera2[i] += a2[j][k]
                centerb1[i] += b1[j][k]
                centerb2[i] += b2[j][k]
                centerx1[i] += x1[j][k]
                centerx2[i] += x2[j][k]
                centery1[i] += y1[j][k]
                centery2[i] += y2[j][k]
                centerD1[i] += D1[j][k]  # 高程
        centerL1[i] /= Count
        centerL2[i] /= Count
        centera1[i] /= Count
        centera2[i] /= Count
        centerb1[i] /= Count
        centerb2[i] /= Count
        centerx1[i] /= Count
        centerx2[i] /= Count
        centery1[i] /= Count
        centery2[i] /= Count
        centerD1[i] /= Count  # 高程

    print("\t[{}] [DoSuperpixel.py]: K-means".format(time.ctime()[11:19]))
    for iteration in range(iterationNum + 1):
        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_1".format(time.ctime()[11:19], iteration))
        for i in range(nRows):
            for j in range(nCols):
                dist[i][j] = DBL_MAX
        for i in range(seedNum):
            # print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_seed{}".format(time.ctime()[11:19], iteration, i))
            x = seedArray[i].x
            y = seedArray[i].y
            minX = int(0 if x - StepX <= 0 else x - StepX)
            minY = int(0 if y - StepY <= 0 else y - StepY)
            maxX = int(nRows - 1 if x + StepX >= nRows - 1 else x + StepX)
            maxY = int(nCols - 1 if y + StepY >= nCols - 1 else y + StepY)

            # my implementation start
            step1_min_x = minX
            step1_max_x = maxX + 1
            step1_min_y = minY
            step1_max_y = maxY + 1
            step1_vpow = np.vectorize(lambda _: _ * _)
            step1_L1_pow = step1_vpow(L1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerL1[i])
            step1_L2_pow = step1_vpow(L2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerL2[i])
            step1_a1_pow = step1_vpow(a1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centera1[i])
            step1_a2_pow = step1_vpow(a2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centera2[i])
            step1_b1_pow = step1_vpow(b1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerb1[i])
            step1_b2_pow = step1_vpow(b2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerb2[i])
            step1_x1_pow = step1_vpow(x1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerx1[i])
            step1_x2_pow = step1_vpow(x2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerx2[i])
            step1_y1_pow = step1_vpow(y1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centery1[i])
            step1_y2_pow = step1_vpow(y2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centery2[i])
            step1_D1_pow = step1_vpow(D1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerD1[i])   # 高程
            step1_D = step1_L1_pow + step1_L2_pow + step1_a1_pow + step1_a2_pow + step1_b1_pow + step1_b2_pow + \
                      step1_x1_pow + step1_x2_pow + step1_y1_pow + step1_y2_pow + step1_D1_pow

            step1_if = (step1_D - dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] < 0).astype(np.uint16)
            step1_neg_if = 1 - step1_if
            new_label[step1_min_x: step1_max_x, step1_min_y: step1_max_y] *= step1_neg_if
            new_label[step1_min_x: step1_max_x, step1_min_y: step1_max_y] += (step1_if * i)

            dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] *= step1_neg_if
            step1_D_to_plus = step1_D * step1_if
            dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] += step1_D_to_plus

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_2".format(time.ctime()[11:19], iteration))
        for i in range(seedNum):
            centerL1[i] = 0
            centerL2[i] = 0
            centera1[i] = 0
            centera2[i] = 0
            centerb1[i] = 0
            centerb2[i] = 0
            centerx1[i] = 0
            centerx2[i] = 0
            centery1[i] = 0
            centery2[i] = 0
            centerD1[i] = 0   # 高程
            WSum[i] = 0
            clusterSize[i] = 0
            seedArray[i].x = 0
            seedArray[i].y = 0

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_3".format(time.ctime()[11:19], iteration))
        label = new_label.copy().reshape([nRows * nCols])

        for i in range(nRows):
            for j in range(nCols):

                L = label[i * nCols + j]  # int
                Weight = W[i][j]  # double
                centerL1[L] += Weight * L1[i][j]
                centerL2[L] += Weight * L2[i][j]
                centera1[L] += Weight * a1[i][j]
                centera2[L] += Weight * a2[i][j]
                centerb1[L] += Weight * b1[i][j]
                centerb2[L] += Weight * b2[i][j]
                centerx1[L] += Weight * x1[i][j]
                centerx2[L] += Weight * x2[i][j]
                centery1[L] += Weight * y1[i][j]
                centery2[L] += Weight * y2[i][j]
                centerD1[L] += Weight * D1[i][j]  # 高程
                clusterSize[L] += 1
                WSum[L] += Weight
                seedArray[L].x += i
                seedArray[L].y += j
        # previous implementation end

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_4".format(time.ctime()[11:19], iteration))
        for i in range(seedNum):
            WSum[i] = 1 if WSum[i] == 0 else WSum[i]
            clusterSize[i] = 1 if clusterSize[i] == 0 else clusterSize[i]

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_5".format(time.ctime()[11:19], iteration))
        for i in range(seedNum):
            centerL1[i] /= WSum[i]
            centerL2[i] /= WSum[i]
            centera1[i] /= WSum[i]
            centera2[i] /= WSum[i]
            centerb1[i] /= WSum[i]
            centerb2[i] /= WSum[i]
            centerx1[i] /= WSum[i]
            centerx2[i] /= WSum[i]
            centery1[i] /= WSum[i]
            centery2[i] /= WSum[i]
            centerD1[i] /= WSum[i]
            seedArray[i].x /= clusterSize[i]
            seedArray[i].y /= clusterSize[i]

    threshold = int((nRows * nCols) / (seedNum * thresholdCoef))
    preEnforceConnectivity(label, nRows, nCols)

    label = EnforceConnectivity(L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, D1, W, label, threshold, nRows, nCols)
    return label
