function H = do_computeHomography(cur_disparity_val,entropyMap1, entropyMap2)
    % 获取当前区域内的2D点和对应的视差
    points2D = cur_disparity_val(:, 1:2);
    disparity = cur_disparity_val(:, 3);
    
    % 生成目标点
    target_points = [points2D(:,1) + disparity, points2D(:,2)];
    
    % 用RANSAC算法计算单应矩阵
    %[H, ~] = ransacHomography(points2D, target_points, 4, 20000, 2.0);
     [H, ~] = ransacHomography2(points2D, target_points, 4, 1000, 2.0,entropyMap1, entropyMap2);

end