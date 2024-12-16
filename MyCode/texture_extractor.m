
function texture_features = texture_extractor(input_data)
    % 输入：input_data - 归一化后的时频数据，大小为 [H, W, C, B]
    % H: 高度（Height）, W: 宽度（Width）, C: 通道数（Channels）, B: 批量大小（Batch size）
    % 输出：texture_features - 提取的纹理特征，大小为 [B, feature_dim]

    % 参数设置
    dilation_rates = [1, 3, 5]; % 膨胀率
    num_filters = 8;          % 每个膨胀卷积层的卷积核数量
    fc_hidden_dim = 8;        % 全连接层的隐藏单元数
    fc_output_dim = 32;        % 最终输出特征维度


    % 获取输入数据的维度
    [H, W, C, B] = size(input_data);

    % 初始化卷积核和偏置
    conv_kernels = cell(1, numel(dilation_rates));
    conv_biases = cell(1, numel(dilation_rates));

    for i = 1:numel(dilation_rates)
        conv_kernels{i} = randn(3, 3, C, num_filters) * 0.01; % [kernel_height, kernel_width, in_channels, out_channels]
        conv_biases{i} = zeros(1, 1, num_filters);            % 偏置
    end

    % 初始化全连接层权重和偏置
    fc1_weights = randn(num_filters * numel(dilation_rates), fc_hidden_dim) * 0.01;
    fc1_bias = zeros(1, fc_hidden_dim);

    fc2_weights = randn(fc_hidden_dim, fc_output_dim) * 0.01;
    fc2_bias = zeros(1, fc_output_dim);

    % ======================
    % 1. 膨胀卷积特征提取
    % ======================
    dcl_features = cell(1, numel(dilation_rates)); % 存储每个膨胀卷积层的输出
    for i = 1:numel(dilation_rates)
        dilation_rate = dilation_rates(i);

        % 膨胀卷积计算
        dcl_output = dilated_conv2d(input_data, conv_kernels{i}, conv_biases{i}, dilation_rate);

        % 激活函数（ReLU）+ 最大池化（2x2）
        dcl_output = max(0, dcl_output); % ReLU
        dcl_output = max_pool2d(dcl_output, 2);

        % 全局平均池化
        dcl_features{i} = global_avg_pool(dcl_output); % [1, 1, num_filters, B]
    end

    % ======================
    % 2. 特征拼接
    % ======================
    % 将所有膨胀卷积层的输出拼接为一个向量
    concatenated_features = cat(3, dcl_features{:}); % [1, 1, num_filters * numel(dilation_rates), B]
    concatenated_features = reshape(concatenated_features, [], B)'; % [B, num_filters * numel(dilation_rates)]

    % ======================
    % 3. 全连接层处理（MNLs）
    % ======================
    % 第一层全连接 + 激活函数（ReLU）
    fc1_output = max(0, concatenated_features * fc1_weights + fc1_bias);

    % 第二层全连接 + 激活函数（ReLU）
    texture_features = max(0, fc1_output * fc2_weights + fc2_bias); % 输出纹理特征
end

% ======================
% 辅助函数
% ======================
function output = dilated_conv2d(input, kernel, bias, dilation_rate)
    % 膨胀卷积实现
    [H, W, C, B] = size(input);  % 输入数据的尺寸
    [kh, kw, kc, num_filters] = size(kernel);  % 卷积核的尺寸
    output = zeros(H, W, num_filters, B);  % 输出的尺寸

    % 膨胀卷积需要通过调整卷积核的步长来实现
    % 通过在卷积核中插入零来实现膨胀
    dilated_kernel = zeros(kh + (kh - 1) * (dilation_rate - 1), ...
                           kw + (kw - 1) * (dilation_rate - 1), kc, num_filters);

    for f = 1:num_filters
        for c = 1:kc
            dilated_kernel(:, :, c, f) = insert_zeros(kernel(:, :, c, f), dilation_rate);
        end
    end

    % 执行卷积
    for b = 1:B
        for f = 1:num_filters
            conv_sum = zeros(H, W);
            for c = 1:C
                % 使用扩展后的卷积核执行常规的二维卷积
                conv_sum = conv_sum + conv2(input(:, :, c, b), dilated_kernel(:, :, c, f), 'same');
            end
            % 加偏置
            output(:, :, f, b) = conv_sum + bias(:, :, f);
        end
    end
end

function dilated_kernel = insert_zeros(kernel, dilation_rate)
    % 将卷积核中插入零来实现膨胀
    [kh, kw] = size(kernel);
    dilated_kernel = zeros(kh + (kh - 1) * (dilation_rate - 1), ...
                           kw + (kw - 1) * (dilation_rate - 1));

    for i = 1:kh
        for j = 1:kw
            dilated_kernel((i - 1) * dilation_rate + 1, (j - 1) * dilation_rate + 1) = kernel(i, j);
        end
    end
end


function output = max_pool2d(input, pool_size)
    [H, W, C, B] = size(input);
    
    % 计算能整除池化窗口的最大尺寸
    new_H = floor(H / pool_size) * pool_size;
    new_W = floor(W / pool_size) * pool_size;

    % 裁剪输入数据
    input_cropped = input(1:new_H, 1:new_W, :, :);

    % 计算池化后的尺寸
    pooled_H = floor(new_H / pool_size);
    pooled_W = floor(new_W / pool_size);
    output = zeros(pooled_H, pooled_W, C, B);

    for b = 1:B
        for c = 1:C
            % 每个通道进行池化
            output(:, :, c, b) = blockproc(input_cropped(:, :, c, b), [pool_size, pool_size], @(block) max(block.data(:)));
        end
    end
end


function output = global_avg_pool(input)
    % 全局平均池化实现
    [H, W, C, B] = size(input);
    output = zeros(1, 1, C, B);

    for b = 1:B
        for c = 1:C
            % 每个通道进行平均
            output(1, 1, c, b) = mean(input(:, :, c, b), 'all');
        end
    end
end

