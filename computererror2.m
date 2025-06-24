function [errorMatrix, largeErrorPoints] = computererror2(modelFile, trueDisparityMapFile)
    close all;
    % 读取之前保存的单应矩阵模型
    load(modelFile, 'model_all');
    % 读取真实视差图
    true_disparity_map = imread(trueDisparityMapFile);
    
    % 处理NaN值
    if any(isnan(true_disparity_map(:)))
        true_disparity_map(isnan(true_disparity_map)) = 0;
    end
    true_disparity_map = double(true_disparity_map);
    
    % 初始化输出矩阵
    errorMatrix = double.empty(0,4);  % [超像素标签, x坐标, y坐标, 误差]
    largeErrorPoints = double.empty(0, 4);  % [x, y, projected_x, error]
    pointsToDelete = [];
    
    % 计算全局统计量用于动态阈值
    valid_disparities = true_disparity_map(true_disparity_map > 0);
    global_median = median(valid_disparities(:));
    global_std = std(valid_disparities(:));
    error_threshold = 3 * global_std;  % 动态阈值
    
    % 进度条
    hWaitBar = waitbar(0, 'Processing superpixels...');
    
    % 遍历所有超像素区域
    num_superpixels = size(model_all, 2);
    for i = 1:num_superpixels
        cur_label = model_all{1, i};
        model_data = model_all{2, i};
        H = model_data{1}; % 单应矩阵
        
        if isempty(H)
            continue;
        end
        
        points2D = model_data{2}(:, 1:2);
        num_points = size(points2D, 1);
        
        % 处理当前超像素中的所有点
        for j = 1:num_points
            selected_point = points2D(j, :);
            
            % 检查点是否在图像边界内
            if ~isValidPoint(selected_point, size(true_disparity_map))
                continue;
            end
            
            % 计算投影点
            projected_point = projectPoint(H, selected_point);
            
            % 获取真实视差值
            true_value = true_disparity_map(round(selected_point(2)), round(selected_point(1)));
            
            % 计算真实点位置和误差
            true_point = [selected_point(1) + true_value, selected_point(2)];
            error_vector = projected_point - true_point;
            
            % 计算误差的不同度量
            error_x = error_vector(1);
            error_y = error_vector(2);
            error_norm = norm(error_vector);
            
            % 检查局部一致性
            local_stats = checkLocalConsistency(true_disparity_map, selected_point, 3);
            
            % 基于多个标准的误差评估
            if isReliableError(error_x, error_y, error_norm, local_stats, error_threshold)
                % 添加到正常误差矩阵
                errorMatrix = [errorMatrix; double(cur_label), ...
                             double(selected_point(1)), ...
                             double(selected_point(2)), ...
                             error_x];
                             
            elseif isLargeButUsefulError(error_x, error_y, error_norm, local_stats, error_threshold)
                % 添加到大误差点集合
                disp = projected_point - selected_point;
                largeErrorPoints = [largeErrorPoints; ...
                                  selected_point(1), ...
                                  selected_point(2), ...
                                  disp(1), ...
                                  error_x];
            else
                % 添加到需要删除的点
                pointsToDelete = [pointsToDelete; selected_point(1), selected_point(2)];
            end
        end
        
        waitbar(i / num_superpixels, hWaitBar, sprintf('Processing superpixel %d/%d', i, num_superpixels));
    end
    
    % 清理误差矩阵
    errorMatrix = cleanErrorMatrix(errorMatrix, pointsToDelete);
    
    % 保存结果
    if ~isempty(errorMatrix)
        save('data2/data01/beiyong/YD_stereo_error.mat', 'errorMatrix');
    end
    
    close(hWaitBar);
end

function valid = isValidPoint(point, imgSize)
    % 检查点是否在图像边界内
    valid = point(1) >= 1 && point(1) <= imgSize(2) && ...
            point(2) >= 1 && point(2) <= imgSize(1);
end

function projected = projectPoint(H, point)
    % 使用单应矩阵投影点
    homog_point = H * [point, 1]';
    homog_point = homog_point / homog_point(3);
    projected = homog_point(1:2)';
end

function stats = checkLocalConsistency(disparity_map, point, window_size)
    % 计算局部区域的统计信息
    y_start = max(1, round(point(2))-window_size);
    y_end = min(size(disparity_map,1), round(point(2))+window_size);
    x_start = max(1, round(point(1))-window_size);
    x_end = min(size(disparity_map,2), round(point(1))+window_size);
    
    neighborhood = disparity_map(y_start:y_end, x_start:x_end);
    
    stats.median = median(neighborhood(:));
    stats.std = std(neighborhood(:));
    stats.mean = mean(neighborhood(:));
end

function reliable = isReliableError(error_x, error_y, error_norm, local_stats, threshold)
    % 判断误差是否可靠
    reliable = abs(error_x) <= threshold && ...
               abs(error_y) <= threshold/2 && ...  % 垂直方向误差应该更小
               error_norm <= threshold * 1.2 && ...
               local_stats.std < threshold/2;  % 局部区域应该相对平滑
end

function useful = isLargeButUsefulError(error_x, error_y, error_norm, local_stats, threshold)
    % 判断较大的误差是否仍然有用
    extended_threshold = threshold * 2;
    useful = abs(error_x) <= extended_threshold && ...
             abs(error_y) <= threshold && ...
             error_norm <= extended_threshold * 1.2 && ...
             local_stats.std < threshold;
end

function cleaned_matrix = cleanErrorMatrix(error_matrix, points_to_delete)
    % 清理误差矩阵中的无效点
    if isempty(points_to_delete)
        cleaned_matrix = error_matrix;
        return;
    end
    
    mask = true(size(error_matrix, 1), 1);
    for k = 1:size(points_to_delete, 1)
        mask = mask & ~(error_matrix(:, 2) == points_to_delete(k, 1) & ...
                       error_matrix(:, 3) == points_to_delete(k, 2));
    end
    cleaned_matrix = error_matrix(mask, :);
end