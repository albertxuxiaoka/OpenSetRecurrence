% 替换 ReLU 为 Swish
function X_out = mnl_mapping(X_in, output_dim)
    hidden_dim = 64;  % 隐藏层维度

    % 确保输入数据是数值型且为列向量
    if isvector(X_in)
        X_in = X_in(:);  % 将输入转化为列向量
    end
    
    % 生成权重矩阵
    W1 = randn(hidden_dim, length(X_in));  % 隐藏层权重矩阵
    W2 = randn(output_dim, hidden_dim);    % 输出层权重矩阵

    % 计算隐藏层的输出
    X_hidden = W1 * X_in;  % 矩阵乘法，得到隐藏层的线性输出

    % 使用 Swish 激活函数
    X_hidden = X_hidden .* (1 ./ (1 + exp(-X_hidden)));  % Swish 激活函数

    % 计算输出层的输出
    X_out = W2 * X_hidden;  % 线性变换得到输出层结果
end
