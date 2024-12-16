function predicted_labels = clustering_with_labels(features, labels, num_classes)
    % 聚类器: 使用 K-Means 对特征进行聚类，并输出预测标签
    % 输入:
    %   features: 特征矩阵 (样本数 x 特征维度)
    %   labels: 真实标签 (样本数 x 1)
    %   num_classes: 类别数量 (K-Means 的聚类数)
    % 输出:
    %   predicted_labels: K-Means 的预测标签 (样本数 x 1)
    
    % 检查输入
    if size(features, 1) ~= length(labels)
        error('特征数量与标签数量不匹配');
    end

    % 使用 K-Means 进行聚类
    % 参数: 'Replicates' 用于增加鲁棒性，尝试多次初始随机种子
    rng(42); % 固定随机数种子，确保结果可重复
    predicted_labels = kmeans(features, num_classes, 'Replicates', 10);

    % 计算聚类结果与真实标签的匹配度
    clustering_accuracy = compute_clustering_accuracy(predicted_labels, labels, num_classes);

    % 输出结果
    fprintf('聚类器的分类准确率: %.2f%%\n', clustering_accuracy * 100);
end

function accuracy = compute_clustering_accuracy(predicted_labels, true_labels, num_classes)
    % 计算聚类准确率 (通过最大化类别映射匹配)
    % 输入:
    %   predicted_labels: K-Means 的预测标签
    %   true_labels: 真实标签
    %   num_classes: 类别数量
    % 输出:
    %   accuracy: 聚类准确率 (0~1)

    % 创建类别混淆矩阵
    conf_matrix = zeros(num_classes, num_classes);
    for i = 1:num_classes
        for j = 1:num_classes
            conf_matrix(i, j) = sum(predicted_labels == i & true_labels == j);
        end
    end

    % 通过最大二分图匹配解决标签重映射问题
    % 使用 Hungarian 算法找到最佳类别映射
    assignment = munkres(-conf_matrix); % munkres 是 Matlab 的 Hungarian 算法实现

    % 计算准确率
    correct_predictions = 0;
    for i = 1:num_classes
        correct_predictions = correct_predictions + conf_matrix(i, assignment(i));
    end
    accuracy = correct_predictions / length(true_labels);
end
