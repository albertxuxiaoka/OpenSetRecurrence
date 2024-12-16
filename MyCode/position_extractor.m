function mapped_features = position_extractor(TFS)
% 假设 TFS 是一个时频图，大小为 M×N（M 为频率点数，N 为时间窗口数）
% 例如，TFS 的大小是 1024×9764，TFS 是从数据集中提取的时频图矩阵    
% 输入时频图的尺寸
    [M, N] = size(TFS);

    % 1. 归一化时频图
    TFS_min = min(TFS(:));
    TFS_max = max(TFS(:));
    TFS_normalized = (TFS - TFS_min) / (TFS_max - TFS_min);

    % 2. 位置编码：为每个频率点和时间窗口生成位置编码
    % 使用正弦和余弦函数来生成位置编码（位置编码公式来自Transformer模型）
    pos_encoding = zeros(M, N); % 初始化位置编码矩阵

    % 为每个频率点和时间窗口计算位置编码
    for m = 1:M
        for n = 1:N
            % 正弦和余弦编码（频率和时间）
            pos_encoding(m, n) = sin(m / (10000^(2 * (m - 1) / M))) + cos(n / (10000^(2 * (n - 1) / N)));
        end
    end

    % 3. 将位置编码加到时频图的每个元素上
    TFS_with_pos_encoding = TFS_normalized + pos_encoding;

    % 4. 位置特征提取：通过自注意力机制来提取位置特征
    % 假设我们使用一个简单的自注意力机制（如基于Query、Key、Value的机制）

    % 定义自注意力层
    Q = TFS_with_pos_encoding; % Query
    K = TFS_with_pos_encoding; % Key
    V = TFS_with_pos_encoding; % Value

    % 自注意力计算
    % 注意力得分计算：Q * K^T
    attention_scores = Q * K'; 

    % 对得分进行缩放并应用softmax
    attention_scores_scaled = attention_scores / sqrt(M);  % 缩放
    attention_weights = softmax(attention_scores_scaled, 2);  % 按行归一化

    % 计算加权值：Attention = Attention_Weights * V
    output_attention = attention_weights * V;

    % 5. 展平自注意力输出并进行 MNL 映射
    % 展平二维输出（M x N）为一维向量
    flattened_output = reshape(output_attention, [], 1);  % 将 M x N 的输出展平为一维向量

    % 假设 flattened_output 是展平后的一维特征向量，长度为 num_features
    num_features = length(flattened_output);  % 一维向量长度
    num_samples = 1;  % 样本数（根据需求调整）

    % 修正后的输入数据和目标数据
    input_data = reshape(flattened_output, num_samples, num_features);  % 输入数据为二维矩阵
    target_data = input_data;  % 自编码任务，目标与输入相同

    % 修正后的网络层定义
    mnl_layers = [
        featureInputLayer(num_features, 'Name', 'input')  % 输入层，指定输入向量的长度

        % 第一个全连接层
        fullyConnectedLayer(8, 'Name', 'fc1')
        reluLayer('Name', 'relu1')

        % 第二个全连接层
        fullyConnectedLayer(8, 'Name', 'fc2')
        reluLayer('Name', 'relu2')

        % 中间降维层：用于提取32维特征
        fullyConnectedLayer(32, 'Name', 'dim_reduction')  % 降维层，维度为32
        reluLayer('Name', 'relu3')

        % 恢复到原始维度以兼容目标数据
        fullyConnectedLayer(num_features, 'Name', 'output')  % 输出与目标数据一致
        regressionLayer('Name', 'regressionOutput')  % 损失层
    ];

    % 配置训练选项
    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'InitialLearnRate', 0.001, ...
        'Verbose', false);

    % 训练网络
    net = trainNetwork(input_data, target_data, mnl_layers, options);

    % 提取降维后的特征
    % 通过访问中间层激活值来获取降维特征
    mapped_features = activations(net, input_data, 'dim_reduction', 'OutputAs', 'rows');

    % 输出降维特征
    disp('降维后的特征（维度为1×32）：');
    disp(mapped_features);
end