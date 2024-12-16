function [X_T_mapped,X_F_mapped] = PE_seperate(TFS)
    % 假设 TFS 是一个时频图，大小为 M×N（M 为频率点数，N 为时间窗口数）
    [M, N] = size(TFS);

    % 1. 归一化时频图
    TFS_min = min(TFS(:));
    TFS_max = max(TFS(:));
    TFS_normalized = (TFS - TFS_min) / (TFS_max - TFS_min);

    % 2. 生成位置编码
    % 定义手工频率 ωw = 10000^(-2w/W)
    W = M; % 频率轴的长度
    omega_w = 10000.^(-2 * (0:(W-1)) / W);

    % 初始化位置编码矩阵
    pos_encoding_time = zeros(M, N);
    pos_encoding_freq = zeros(M, N);

    % 计算位置编码
    for m = 1:M
        for n = 1:N
            % 时域位置编码（加位置编码到转置矩阵）
            pos_encoding_time(m, n) = sin(m / omega_w(mod(m-1, W) + 1)) + cos(n / omega_w(mod(n-1, W) + 1));
            % 频域位置编码（加位置编码到原始矩阵）
            pos_encoding_freq(m, n) = sin(n / omega_w(mod(n-1, W) + 1)) + cos(m / omega_w(mod(m-1, W) + 1));
        end
    end

    % 3. 添加位置编码并分别提取时域和频域特征
    X_T = TFS_normalized' + pos_encoding_time';  % 时域位置特征
    X_F = TFS_normalized + pos_encoding_freq;   % 频域位置特征



    % 时域和频域的自注意力特征提取
    X_T_sa = self_attention(X_T);
    X_F_sa = self_attention(X_F);

    % 5. 展平并通过 MNL 映射
    % 展平为一维向量
    X_T_flatten = reshape(X_T_sa, [], 1);
    X_F_flatten = reshape(X_F_sa, [], 1);



    % 将展平后的特征通过 MNL 映射
    X_T_mapped = mnl_mapping(X_T_flatten, 32); % 映射到 32 维
    X_F_mapped = mnl_mapping(X_F_flatten, 32); % 映射到 32 维

    % 输出映射后的时域和频域位置特征
    disp('时域位置特征 (降维后)：');
    disp(X_T_mapped);

    disp('频域位置特征 (降维后)：');
    disp(X_F_mapped);
end
