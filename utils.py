from osgeo import gdal
import cv2
import numpy as np
import matplotlib.image as imgplt
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def create_preview(dataset, target_path, red=0, green=1, blue=2):
    """
    生成遥感影像的预览图
    :param dataset: 影像数据
    :param red: 预览图红色采用影像的波段索引
    :param green: 预览图绿色采用影像的波段索引
    :param blue: 预览图蓝色采用影像的波段索引
    :param target_path: 预览图存储地址
    :return:
    """
    band_count = dataset.RasterCount
    cols = dataset.RasterXSize  # 列数
    rows = dataset.RasterYSize  # 行数
    im_data = dataset.ReadAsArray(0, 0, cols, rows)
    # 多波段影像根据红绿蓝索引合成图片
    if band_count >= 3:
        band_red = im_data[red]
        data_red = translate(band_red, cols, rows)
        band_green = im_data[green]
        data_green = translate(band_green, cols, rows)
        band_blue = im_data[blue]
        data_blue = translate(band_blue, cols, rows)

        x = np.zeros([rows, cols, 3], dtype=np.uint8)
        for row_index in range(rows):
            for col_index in range(cols):
                x[row_index][col_index][0] = data_red[row_index][col_index]
                x[row_index][col_index][1] = data_green[row_index][col_index]
                x[row_index][col_index][2] = data_blue[row_index][col_index]
        cv2.imwrite(target_path, x)
    # 单波段影像直接合成灰度
    else:
        # band = dataset.GetRasterBand(1)
        band = dataset.ReadAsArray(0, 0, cols, rows)
        cv2.imwrite(target_path, translate(band, cols, rows))


def translate(band, cols, rows):
    """将一个波段的值转到0-255的区间"""
    # band_data = band.ReadAsArray(0, 0, cols, rows)
    min = np.min(band)
    max = np.max(band)
    # 以下两行代码band_data处理结果都可以用，一种是将图像rgb值限制在255以下，一种是限制在0到255之间
    # band_data = band / max * 255
    band_data = (band - min) * 255 / (max - min)
    return band_data.astype(np.uint8)


# 多波段影像
# 读取TIFF图像的路径
dataset = gdal.Open("./datas/datas/data01/pair_2/HY_disparity_0.tiff")
# 生成png图像的保存路径
target_path = "./test_fused_images_v1.png"

create_preview(dataset, target_path)

pic = imgplt.imread(target_path)
plt.imshow(pic)
plt.show()
