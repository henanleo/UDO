import cv2
import numpy as np
import skimage.segmentation as seg
from PIL import Image
import disparity2
import disparity
def read_image_with_pillow(filepath):
    pil_image = Image.open(filepath)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 使用 Pillow 读取立体图像并转换为 OpenCV 格式
# left_img = read_image_with_pillow('../02_LSCplus_Superpixel/datas/datas/data01/pair_2/HY_left_0.tiff')
#
# # 检查图像是否读取成功
# if left_img is None:
#     raise IOError("Could not read the left image.")
#
# # 对左图进行直方图拉伸
# left_img_stretched = enhance_contrast(left_img)
# # 确保图像是 uint8 类型，并且范围在 [0, 255] 之间
# left_img_stretched = np.clip(left_img_stretched, 0, 255)
left_img_stretched = read_image_with_pillow('./dataset/handle_YD_left_455.tiff')

# 如果图像是单通道的，将其转换为三通道的图像
if len(left_img_stretched.shape) == 2 or left_img_stretched.shape[2] == 1:
    left_img_stretched = cv2.cvtColor(left_img_stretched, cv2.COLOR_GRAY2BGR)
# 使用 Pillow 读取预先计算的视差图
disparity_image = Image.open('./dataset/hmsmhandle_YD_left_455.tiff')
disparity_map = np.array(disparity_image, dtype=np.float32)  # 读取 16-bit 深度


# disparity_image = read_image_with_pillow('test_fused_images_v1.png')
# disparity_image=np.clip(disparity_image, 0, 255)
# disparity_image = np.nan_to_num(disparity_image, nan=0)  # 将NaN值替换为0
# disparity_map = np.asarray(disparity_image, dtype=np.uint8)
gray_img = cv2.cvtColor(left_img_stretched, cv2.COLOR_BGR2GRAY)
height, width = disparity_map.shape
    # 将图像变换为目标尺寸
img_Ir = cv2.resize(gray_img, (width, height))
# img_disp = cv2.resize(disparity_map, (width, height))
# 执行算法
print('Performing the proposed algorithm ...')
name = 'YD4552'
label_2D = disparity.run(img_Ir, disparity_map, name)

print('DONE ！')
