import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import LSC_Plus_Func
import imtools
import scipy.io as scio
import do_virtualSurfaces as DV
import DSMtools


def do_calPt2Plane(M, pts):

    ABC = M[0:3]
    D = M[3]
    # norm2 = np.linalg.norm(ABC, axis=0)  # 按列向量计算范数

    # 本来应该除以ABC的二阶范数的（平方和开根号），但是ABC的二阶范数恒为一，省略分母
    Dist = np.abs(np.dot(ABC, pts) + D)
    return Dist


def do_calculateDSMValue(model, pt_XYZs):
    """通过虚拟plane模型约束dsm融合计算
    输入：
    model：当前类的虚拟plane模型
    pt_XYZs：当前点的50个3D坐标
    输出：
    dsm_val：当前点的DSM值"""

    # 选取比例（ratio越小，获得的dsm_val却贴近虚拟表面）
    # ratio = 0.4

    # 计算点到虚拟plane之间的距离
    dist = do_calPt2Plane(model, pt_XYZs)
    pt_XYZD = np.vstack((pt_XYZs, dist))

    # 过滤NaN
    pt_XYZD_1 = imtools.do_remove_nan(pt_XYZD)

    # 按照距离对矩阵排序
    sorted_pt_XYZD = pt_XYZD_1[:, pt_XYZD_1[3].argsort()]

    # 按照ratio获取靠前的列数
    NUM = int(np.ceil(ratio * np.size(sorted_pt_XYZD, 1)))

    # # 获取前NUM的DSM中值
    # dsm_val = np.median(sorted_pt_XYZD[2, 0:NUM])

    # method2-获取前NUM的DSM值
    dsm_val = sorted_pt_XYZD[2, 0:NUM]
    return dsm_val


def do_planeModelMedian(dsm_3D, class_model, class_pts, class_ind):
    """本函数实现：基于当前类模型的约束实现中值计算
    输入：
    dsm_3D, 50层的DSM
    class_model, 当前类模型
    class_pts, 当前类作用点
    class_ind，当前类作用点的索引
    """
    class_dsm = np.zeros(class_ind.shape)

    # 逐个点计算
    pts_num = np.size(class_pts, 0)
    pts_Y = class_pts[:, 1] - 1  # Y坐标，MATLAB坐标从1开始，需要减去1
    pts_X = class_pts[:, 0] - 1  # X坐标
    xy_pts = np.array([pts_X, pts_Y])  # class内平面坐标
    pts_Zs = dsm_3D[class_ind, :]  # 当前类内点的Z坐标，row代表50层，col代表类内点
    cur_class_DSMVal = np.zeros(1)

    # 类内点循环，每个点都有50个Z数值
    for i in range(pts_num):
        cur_pt_x = pts_X[i]
        cur_pt_y = pts_Y[i]

        cur_pt_Xs = np.ones(50)*cur_pt_x
        cur_pt_Ys = np.ones(50)*cur_pt_y
        cur_pt_Zs = pts_Zs[i, :]

        cur_pt_XYZs = np.array([cur_pt_Xs, cur_pt_Ys, cur_pt_Zs])

        # 通过虚拟plane模型约束dsm融合计算
        cur_pt_DSMVals = do_calculateDSMValue(class_model, cur_pt_XYZs)

        # ---------------------两种取值的方法---------------------
        # method1-将融合后的dsm值赋值给当前点的位置
        # cur_pt_val = np.median(cur_pt_DSMVals)
        # class_dsm[int(cur_pt_y), int(cur_pt_x)] = cur_pt_val

        # method2-存储类内点的所有dsm值
        cur_class_DSMVal = np.append(cur_class_DSMVal, cur_pt_DSMVals)

    # method2-求类内所有点的中值
    class_dsm_med = np.median(cur_class_DSMVal[1:])
    class_dsm[class_ind] = class_dsm_med

    return class_dsm


def do_calSupOutDSM(supixel_out, dsm_3D, all_dsm_median):
    """本函数实现：求聚类剩下的点的中值
    输入：
    输出："""

    cur_out_ind = supixel_out == 1  # 遗漏区域的bool

    cur_out_dsm = dsm_3D[cur_out_ind, :]  # 遗漏区域的DSM

    # 过滤nan
    ind = ~np.isnan(cur_out_dsm)
    data_without_nan = cur_out_dsm[ind]

    # median
    med_dsm = np.median(data_without_nan)

    all_dsm_median[cur_out_ind] = med_dsm

    return all_dsm_median


def do_propRegionalMedian(label_2D, dsm_3D, model_all):
    """对每个label执行中值滤波"""

    min_h = 10   # 最小高程
    max_h = 50   # 最大高程
    rows, cols, zs = dsm_3D.shape

    # 生成存储计算结果的存储器
    all_dsm_median = np.zeros((rows, cols), dtype="float64")
    # all_dsm_median = dsm_3D[:, :, 0]

    label_uni = np.unique(label_2D)
    label_num = len(label_uni)

    # 超像素循环
    for i in tqdm(range(label_num)):

        # print('%d', i)
        cur_supixel_ind = label_2D == label_uni[i]      # 当前超像素cover

        cur_supixel_class_ind = np.zeros((rows, cols))  # 当前超像素内的类别cover

        model2pts = model_all[1, i]

        class_num = np.size(model2pts, 1)   # 聚类数

        # 类循环
        for j in range(class_num):
            cur_class_model = model2pts[0, j][0]  # 当前类模型
            cur_class_pts = model2pts[1, j]       # 当前类作用点坐标。第一列：X坐标（col）；第二列：Y坐标（row）。从1开始
            cur_class_bw = model2pts[2, j]
            cur_class_ind = cur_class_bw == 1     # 当前类对应区域的索引

            cur_class_dsm = do_planeModelMedian(dsm_3D, cur_class_model, cur_class_pts, cur_class_ind)

            # 将当前类的DSM存储进来
            all_dsm_median = all_dsm_median + cur_class_dsm
            cur_supixel_class_ind = cur_supixel_class_ind + cur_class_ind

        # 当前超像素内遗漏区域DSM融合
        cur_supixel_out = cur_supixel_ind - cur_supixel_class_ind  # 当前超像素遗漏区域

        # 补充遗留区域的DSM
        all_dsm_median = do_calSupOutDSM(cur_supixel_out, dsm_3D, all_dsm_median)

    return all_dsm_median


def do_imageRegistration(img, dsm_seg):
    """本函数调用MATLAB程序实现"""

    img_Ir_r1 = scio.loadmat('01_Registering/results/img_Ir_r1.mat')
    img_Ir = img_Ir_r1['img_Ir']
    return img_Ir


def do_imageRegistration_R01(img, dsm_seg):
    """本函数调用MATLAB程序实现"""

    img_Ir_r01 = scio.loadmat('01_Registering/results_R02A/img_Ir_R02A.mat')
    img_Ir = img_Ir_r01['img_Ir']
    return img_Ir


def do_imageRegistration_R02A(img, dsm_seg):
    """本函数调用MATLAB程序实现"""

    img_Ir_r02A = scio.loadmat('01_Registering/results_R02A/img_Ir_R02A.mat')
    img_Ir = img_Ir_r02A['img_Ir']
    return img_Ir


def do_imageRegistration_R02B(img, dsm_seg):
    """本函数调用MATLAB程序实现"""

    img_Ir_r02B = scio.loadmat('01_Registering/results_R02B/img_Ir_R02B.mat')
    img_Ir = img_Ir_r02B['img_Ir']
    return img_Ir


def do_imageRegistration_R02C(img, dsm_seg):
    """本函数调用MATLAB程序实现"""

    img_Ir_r02C = scio.loadmat('01_Registering/results_R02C/img_Ir_R02C.mat')
    img_Ir = img_Ir_r02C['img_Ir']
    return img_Ir


def do_imageRegistration_R02D(img, dsm_seg):
    """本函数调用MATLAB程序实现"""

    img_Ir_r02D = scio.loadmat('01_Registering/results_R02D/img_Ir_R02D.mat')
    img_Ir = img_Ir_r02D['img_Ir']
    return img_Ir


