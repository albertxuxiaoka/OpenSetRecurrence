
    %% 参数设置
    batch_size = 90;
    texture_dim = 32;       % 纹理特征维度
    position_dim = 32;      % 位置特征维度
    semantic_dim = 256;     % 语义特征维度
    num_classes = 9;        % 类别数量
    eta1 = 0.5;             % 中心损失权重
    eta2 = 0.5;             % 聚类损失权重
    eta3 = 1;             % 交叉熵损失权重
    alpha = 0.5;
    
    % 模拟特征输入和标签
    texture_feat = texture_feature;     % 纹理特征
    time_feat = time_feature;       % 时间特征
    freq_feat = frequency_feature; % 频率特征
    %labels = labels;   % 随机生成类别标签
    
    %% 特征拼接和语义映射
    semantic_feat = semantic_mapping(texture_feat, time_feat, freq_feat, semantic_dim);
    
    %% 损失计算
    [center_loss, cluster_loss, cross_entropy_loss] = compute_losses(...
        semantic_feat, labels, num_classes, semantic_dim);
    total_loss = eta1 * center_loss + eta2 * cluster_loss + eta3 * cross_entropy_loss;
    
    %% 输出结果
    fprintf('Center Loss: %.4f\n', center_loss);
    fprintf('Cluster Loss: %.4f\n', cluster_loss);
    fprintf('Cross-Entropy Loss: %.4f\n', cross_entropy_loss);
    fprintf('Total Loss: %.4f\n', total_loss);


%% 特征拼接并映射到语义空间
function semantic_feat = semantic_mapping(texture_feat, time_feat, freq_feat, semantic_dim)
    % 特征拼接
    combined_feat = normalize([texture_feat, time_feat, freq_feat]);

    
    % 随机初始化权重和偏置 (模拟全连接层)
    input_dim = size(combined_feat, 2);
    W1 = randn(input_dim, semantic_dim) * sqrt(2 / input_dim);
    b1 = randn(1, semantic_dim) * 0.1;         % 偏置向量
    
    % 映射到语义空间
    semantic_feat = max(0, combined_feat * W1 + b1); % ReLU激活函数
end

%% 损失计算
function [center_loss, cluster_loss, cross_entropy_loss] = compute_losses(semantic_feat, labels, num_classes, semantic_dim)
    %% 中心损失 (Center Loss)
    % 随机初始化类别中心
    centers = randn(num_classes, semantic_dim);
    
    
    % 计算每个样本与其类别中心的距离
    center_loss = 0;
    batch_size = size(semantic_feat, 1);
    for i = 1:batch_size
        label = labels(i);
        center_loss = center_loss + norm(semantic_feat(i, :) - centers(label, :))^2;
        fprintf("center_loss:%f\n", center_loss);
        centers(label, :) = centers(label, :) + 0.5 * (semantic_feat(i, :) - centers(label, :));
    end
    center_loss = center_loss / batch_size;
    
    %% 聚类损失 (Cluster Loss)
    margin = 1.0; % 最小类别中心间隔
    cluster_loss = 0;
    for i = 1:num_classes
        for j = i+1:num_classes
            dist = norm(centers(i, :) - centers(j, :));
            cluster_loss = cluster_loss + max(0, margin - dist);
            fprintf("cluster_loss: %f\n",cluster_loss);
            margin = margin * (1 + 0.05);
        end
    end
    
    %% 交叉熵损失 (Cross-Entropy Loss)
    % 随机初始化分类器权重 (模拟全连接层)
    W2 = randn(semantic_dim, num_classes) * sqrt(2 / semantic_dim);
    logits = semantic_feat * W2; % 线性输出
    probabilities = softmax(logits, 2); % Softmax转换为概率分布
    
    % 计算交叉熵
    cross_entropy_loss = 0;
    for i = 1:batch_size
        cross_entropy_loss = cross_entropy_loss - log(probabilities(i, labels(i)));
        fprintf("cross_entropy_loss: %f\n",cross_entropy_loss);
    end
    cross_entropy_loss = cross_entropy_loss / batch_size;
end

%% Softmax 函数
function prob = softmax(logits, dim)
    temperature = 2.0;
    logits = logits / temperature;
    exp_logits = exp(logits - max(logits, [], dim)); % 防止溢出
    prob = exp_logits ./ sum(exp_logits, dim);
end
