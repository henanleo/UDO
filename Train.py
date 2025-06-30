import os

import cv2
import h5py
import numpy as np
from PIL import Image
import subprocess
import scipy.io as sio
from scipy.ndimage import gaussian_filter, median_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
def suppress_extreme_values(disparity_map, percentile=99):

    lower_bound = np.quantile(disparity_map, 1 - percentile / 100)
    upper_bound = np.quantile(disparity_map, percentile / 100)

    disparity_map = np.clip(disparity_map, lower_bound, upper_bound)
    return disparity_map

class DisparityOptimizer:
    def __init__(self, data_path=''):
        self.data_path = data_path
        self.iteration_history = {}
        self.edge_points = None
        self.debug_regions = []

        # 去噪参数
        self.bilateral_params = {
            'sigma_spatial': 1.5,
            'sigma_range': 0.05,
            'window_size': 3
        }

    def segment_building_regions(self, disparity_map, rgb_image=None):
        """分割潜在建筑区域"""
        # 基于视差的基本分割
        from sklearn.cluster import KMeans

        # 展平并准备视差进行聚类
        flat_disp = disparity_map.reshape(-1, 1)
        # 寻找自然聚类（复杂场景需要更多聚类）
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flat_disp)
        labels = kmeans.labels_.reshape(disparity_map.shape)

        # 查找哪些聚类可能是建筑（通常具有较高视差）
        cluster_means = [np.mean(disparity_map[labels == i]) for i in range(n_clusters)]
        building_clusters = [i for i, mean in enumerate(cluster_means) if mean > np.median(cluster_means)]

        # 为潜在建筑区域创建掩码
        building_mask = np.zeros_like(disparity_map, dtype=bool)
        for cluster in building_clusters:
            building_mask = np.logical_or(building_mask, labels == cluster)

        return building_mask
    def bilateral_filter(self, disparity_map, error_map=None):
        """双边滤波"""
        rows, cols = disparity_map.shape
        filtered = np.zeros_like(disparity_map)
        window_size = self.bilateral_params['window_size']
        sigma_s = self.bilateral_params['sigma_spatial']
        sigma_r = self.bilateral_params['sigma_range']

        # 计算置信度
        confidence = None
        if error_map is not None:
            confidence = np.exp(-np.abs(error_map))

        for i in range(rows):
            i_min = max(0, i - window_size)
            i_max = min(rows, i + window_size + 1)
            for j in range(cols):
                j_min = max(0, j - window_size)
                j_max = min(cols, j + window_size + 1)

                window = disparity_map[i_min:i_max, j_min:j_max]

                # 计算空间权重
                y, x = np.mgrid[i_min - i:i_max - i, j_min - j:j_max - j]
                spatial_weight = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))

                # 计算值域权重
                value_weight = np.exp(-np.abs(window - disparity_map[i, j]) / sigma_r)

                # 合并权重
                weight = spatial_weight * value_weight

                if confidence is not None:
                    conf_window = confidence[i_min:i_max, j_min:j_max]
                    weight *= conf_window

                # 归一化和加权平均
                weight_sum = np.sum(weight)
                if weight_sum > 0:
                    filtered[i, j] = np.sum(window * weight) / weight_sum
                else:
                    filtered[i, j] = disparity_map[i, j]

        return filtered

    def denoise_disparity(self, disparity_map, error_matrix, edge_mask):
        """综合去噪处理"""
        # 构建误差图

        error_map = np.zeros_like(disparity_map)
        for entry in error_matrix:
            y = int(entry[2]) - 1
            x = int(entry[1]) - 1
            error_map[y, x] = entry[3]

        filtered_map = disparity_map.copy()
        non_edge_mask = ~edge_mask
        # non_edge_mask 是布尔矩阵，转换为 0/255 格式
        non_edge_mask_uint8 = (non_edge_mask * 255).astype(np.uint8)

        # 保存为图片
        cv2.imwrite("non_edge_mask.png", non_edge_mask_uint8)

        # 仅更新非边缘区域
        bilateral_filtered = self.bilateral_filter(filtered_map, error_map)
        filtered_map[non_edge_mask] = bilateral_filtered[non_edge_mask]
        # 3. 边缘区域特殊处理
        edge_window_size = 5
        edge_coords = np.where(edge_mask)
        for y, x in zip(*edge_coords):
            # 提取局部窗口
            y_start = max(0, y - edge_window_size // 2)
            y_end = min(disparity_map.shape[0], y + edge_window_size // 2 + 1)
            x_start = max(0, x - edge_window_size // 2)
            x_end = min(disparity_map.shape[1], x + edge_window_size // 2 + 1)

            # 计算局部区域的梯度
            window = filtered_map[y_start:y_end, x_start:x_end]
            grad_y, grad_x = np.gradient(window)
            grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # 基于梯度的自适应权重
            weights = 1 / (1 + grad_magnitude)
            weights[window == filtered_map[y, x]] *= 2  # 增加相似值的权重

            # 加权平均
            filtered_map[y, x] = np.average(window, weights=weights)

        return filtered_map

    def detect_edge_regions(self, disparity_map, threshold=8):  # 降低阈值
        from scipy import ndimage
        # 添加高斯平滑预处理
        smooth_disp = gaussian_filter(disparity_map, sigma=1)
        gradient_x = ndimage.sobel(smooth_disp, axis=1)
        gradient_y = ndimage.sobel(smooth_disp, axis=0)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        edge_mask = gradient_magnitude > threshold
        # 添加形态学操作使边缘更连续
        from scipy.ndimage import binary_dilation
        edge_mask = binary_dilation(edge_mask, iterations=1)
        return edge_mask

    def track_region_optimization(self, region_coords, current_disparity, iteration):
        """跟踪特定区域的优化过程"""
        y, x = region_coords
        if (y, x) not in self.iteration_history:
            self.iteration_history[(y, x)] = []
        self.iteration_history[(y, x)].append(current_disparity[y, x])

    def analyze_optimization_direction(self, error_value, current_disp, prev_disp, ground_truth=None):
        """分析优化方向是否正确"""
        if ground_truth is not None:
            current_error = abs(current_disp - ground_truth)
            prev_error = abs(prev_disp - ground_truth)
            is_improving = current_error < prev_error
        else:
            # 如果没有真实值，使用误差变化趋势判断
            is_improving = abs(error_value) < abs(prev_disp - current_disp)

        return is_improving

    def adaptive_weight(self, error_value, iteration, error_matrix):
        base_weight = 0.7

        # 降低基础权重
        error_std = np.median(np.abs(error_matrix[:, 3] - np.median(error_matrix[:, 3])))
        error_std = max(error_std, 0.1)


        # 更保守的置信度计算
        confidence = 1 / (1 + np.abs(error_value) / (error_std + 1e-6))

        # 更慢的迭代衰减
        iteration_factor = 1 / (1 + 0.2 * iteration)

        return base_weight * confidence * iteration_factor

    def plot_optimization_history(self, region_coords):
        """绘制优化历史"""
        plt.figure(figsize=(10, 6))
        values = self.iteration_history[region_coords]
        plt.plot(values, marker='o')  
        plt.title(f'Optimization History for Region {region_coords}')
        plt.xlabel('Iteration')
        plt.ylabel('Disparity Value')
        plt.grid(True)
        plt.savefig(os.path.join(self.data_path, f'optimization_history_{region_coords}.png'))
        plt.close()

    def optimize_disparity(self, max_iterations=1):
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}/{max_iterations}")

            # 运行MATLAB脚本
            try:
                subprocess.run(['matlab', '-batch', "Homography"], check=True)
                subprocess.run(['matlab', '-batch', "errorcomputer"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running MATLAB script: {e}")
                continue
            # 加载数据
            with h5py.File(os.path.join(self.data_path, 'error.mat'), 'r') as f:
                error_matrix = np.array(f['errorMatrix']).T
            print("X range:", np.min(error_matrix[:, 1]), np.max(error_matrix[:, 1]))
            print("Y range:", np.min(error_matrix[:, 2]), np.max(error_matrix[:, 2]))
            if error_matrix.size == 0:
                print("Error matrix is empty. Ending loop.")
                break

            # 加载视差图
            disparity_img = Image.open(os.path.join(self.data_path, 'disp.tiff'))
            disparity_map = np.array(disparity_img)
            old_disparity = disparity_map.copy()
            disparity_img.close()

            # 获取图像尺寸
            height, width = disparity_map.shape
            error_map = np.zeros((height, width), dtype=np.float32)
            weight_map = np.zeros((height, width), dtype=np.float32)

            # 检测边缘区域
            edge_mask = self.detect_edge_regions(disparity_map)

            # 更新视差图
            print("正在更新视差图")
            for idx, entry in enumerate(tqdm(error_matrix)):
                # 确保索引在有效范围内
                y = int(entry[2]) - 1
                x = int(entry[1]) - 1
                error_value = entry[3]


                # 计算权重
                weight = self.adaptive_weight(error_value, iteration, error_matrix)

                # 安全检查边缘标记
                if y < edge_mask.shape[0] and x < edge_mask.shape[1] and edge_mask[y, x]:
                    weight *= 0.5  # 在边缘区域进一步减小步长
                error_map[y, x] = error_value
                weight_map[y, x] = weight
                # 更新视差值
                old_value = disparity_map[y, x]
                delta = error_value * weight
                delta = np.clip(delta, -1.5, 1.5)  # 进一步限制步长

                # 检查优化方向
                new_value = old_value + delta
                disparity_map[y, x] = new_value


            # 应用去噪处理
            print("正在进行去噪处理")
            disparity_map = self.denoise_disparity(disparity_map, error_matrix, edge_mask)
            delta_mean = np.mean(np.abs(old_disparity - disparity_map))
            # 归一化误差图和权重图
            norm_error = 255 * (error_map - np.min(error_map)) / (np.ptp(error_map) + 1e-8)
            norm_weight = 255 * (weight_map - np.min(weight_map)) / (np.ptp(weight_map) + 1e-8)

            # 权重图和误差图保存为PNG
            Image.fromarray(norm_error.astype(np.uint8)).save(
                os.path.join(self.data_path, f'error_map_iter{iteration}.png'))
            Image.fromarray(norm_weight.astype(np.uint8)).save(
                os.path.join(self.data_path, f'weight_map_iter{iteration}.png'))

            print("误差图和权重图已保存")
            print(f"Iteration {iteration} mean delta: {delta_mean}")
            # 保存结果
            updated_disparity_img = Image.fromarray(disparity_map)
            updated_disparity_img.save(os.path.join(self.data_path, 'disp.tiff'))
            print("保存成功")

    def generate_optimization_report(self, iteration, old_disp, new_disp, error_matrix):
        """生成每次迭代的优化报告"""
        report_path = os.path.join(self.data_path, f'optimization_report_{iteration}.txt')
        with open(report_path, 'w') as f:
            f.write(f"=== Optimization Report for Iteration {iteration} ===\n")
            f.write(f"Mean Error: {np.mean(error_matrix[:, 3]):.4f}\n")
            f.write(f"Error Std: {np.std(error_matrix[:, 3]):.4f}\n")
            f.write(f"Max Update: {np.max(np.abs(new_disp - old_disp)):.4f}\n")
            f.write(f"Mean Update: {np.mean(np.abs(new_disp - old_disp)):.4f}\n")

            # 记录边缘区域的变化
            edge_changes = new_disp[self.detect_edge_regions(old_disp)] - old_disp[self.detect_edge_regions(old_disp)]
            f.write(f"Edge Region Mean Change: {np.mean(np.abs(edge_changes)):.4f}\n")

    def check_convergence(self, old_disp, new_disp, threshold=0.1):
        """检查是否收敛"""
        diff = np.abs(old_disp - new_disp)
        return np.mean(diff) < threshold



# 使用示例
optimizer = DisparityOptimizer()
optimizer.optimize_disparity(max_iterations=5)
