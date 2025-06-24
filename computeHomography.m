function H = computeHomography(points1, points2)
    % 使用最小二乘法计算单应矩阵
    points1 = double(points1);
    points2 = double(points2);
    A = [];
    for i = 1:size(points1, 1)
        x = points1(i, 1);
        y = points1(i, 2);
        xp = points2(i, 1);
        yp = points2(i, 2);
        A = [A; -x, -y, -1, 0, 0, 0, x*xp, y*xp, xp];
        A = [A; 0, 0, 0, -x, -y, -1, x*yp, y*yp, yp];
    end
    [~, ~, V] = svd(A);
    H = reshape(V(:, end), 3, 3)';
end
