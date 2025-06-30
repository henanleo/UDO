from osgeo import gdal
import numpy as np
import os
from tqdm import tqdm
from tqdm._tqdm import trange
import pandas as pd
import math
import matplotlib.pyplot as plt


class GRID:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)       # 打开文件

        im_width = dataset.RasterXSize    # 栅格矩阵的列数
        im_height = dataset.RasterYSize   # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

        del dataset
        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")            # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)              # 写入仿射变换参数
        dataset.SetProjection(im_proj)                    # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset


def do_remove_nan(input_data):
    """remove nan in input_data"""

    # 确定位置
    ind = ~np.isnan(input_data[:, 1]).T
    data_without_nan = input_data[ind, :]

    # print(input_data)
    # print(data_without_nan)
    return data_without_nan


def do_remove_outliers(input_data, min_h, max_h):
    """remove otliers in input_data"""
    data_new = input_data[input_data > min_h]
    data_new1 = data_new[data_new < max_h]

    return data_new1


def dsm_Stretch2Uint8(dsm):
    """将DSM中的值拉伸到uint8尺度上"""

    rows, cols = dsm.shape
    dsm_new = np.zeros((rows, cols), dtype="uint8")

    dsm[np.isnan(dsm[:, :])] = 0  # 令所有NAN为0
    min_num = np.min(dsm)
    dsm_ = dsm - min_num

    max_num = np.max(dsm_)
    bei_num = 255/max_num  # 放大尺度
    dsm_new = dsm_ * bei_num  # 按照尺度拉伸

    # plt.imshow(dsm_new)
    # plt.show()
    return dsm_new
