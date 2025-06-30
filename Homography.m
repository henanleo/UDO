close all;

seg_data = load('L_true175_LSCplus.mat'); % 加载分割数据
disparity_map = imread('disp.tiff');%视差图
image1 = imread('left.tiff');%左图
image2 = imread('right.tiff');%右图
windowSize = 15; % 可以根据需要调整

% 计算熵图
entropyMap1 = calculateEntropyMap(image1, windowSize);
entropyMap2 = calculateEntropyMap(image2, windowSize);
disparity_map = double(disparity_map);
disp(fieldnames(seg_data));
supPixel_2DLabel = seg_data.label_2D;        % 2D label

label_1D = unique(unique(supPixel_2DLabel)); % 1D label
sz = size(supPixel_2DLabel);                 % 超像素尺寸
num_label = length(label_1D);    
model_all = cell(2, num_label);  % 存储每个superpixel的模型

ifShow = 0;  % 显示图像开关
hWaitBar = waitbar(0, 'Processing ...');
for i = 1 : num_label
    
    cur_label = label_1D(i);
    cur_BW = supPixel_2DLabel == cur_label;
    if ifShow
        figure, imshow(cur_BW);
    end
   
    % 获取当前区域内的视差值
    cur_disparity_val = double(do_getDisparityVal(cur_BW, disparity_map));  % 当前区域的3D坐标

    % 计算单应矩阵
    %H = do_computeHomography(cur_disparity_val);
    valid_points = ~isnan(cur_disparity_val(:, 1)) & ~isnan(cur_disparity_val(:, 2));
    cur_disparity_val = cur_disparity_val(valid_points, :);
    H = do_computeHomography(cur_disparity_val,entropyMap1, entropyMap2);
    model2pts_ind{1} = H;
    model2pts_ind{2} = cur_disparity_val;
    model2pts_ind{3} = cur_BW;
    model_all{1, i} = cur_label;      % 超像素label
    model_all{2, i} = model2pts_ind;  % 单应矩阵-点坐标-点索引
    % 更新进度条
    waitbar(i / num_label, hWaitBar);
end

save('data\175_model_all.mat', 'model_all');  % 保存单应矩阵模型
computererror('./data/175_model_all.mat', 'disp.tiff');

fprintf('DONE!');
