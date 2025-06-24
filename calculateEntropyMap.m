function entropyMap = calculateEntropyMap(image, windowSize)
    % 确保输入图像是灰度图
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % 获取图像尺寸
    [rows, cols] = size(image);
    
    % 初始化熵图，与原图大小相同，但初始化为0
    entropyMap = zeros(rows, cols);
    
    % 定义滑动窗口的边界
    halfWin = floor(windowSize / 2);
    
    % 遍历图像的每个像素（除了边缘）
    for i = 1+halfWin : rows-halfWin
        for j = 1+halfWin : cols-halfWin
            % 提取当前窗口的灰度值
            window = image(i-halfWin:i+halfWin, j-halfWin:j+halfWin);
            
            % 计算窗口内灰度值的直方图
            histCounts = histcounts(window(:), 256); % 256个灰度级
            
            % 归一化直方图以计算概率
            p = histCounts / sum(histCounts);
            
            % 忽略概率为0的情况（对数运算中的0）
            p(p == 0) = eps; % 使用一个很小的数代替0
            
            % 计算熵
            entropy = -sum(p .* log2(p));
            
            % 将计算得到的熵赋值给熵图的中心像素
            entropyMap(i, j) = entropy;
        end
    end
end
