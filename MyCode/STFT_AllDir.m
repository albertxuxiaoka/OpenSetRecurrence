% 参数设置
fs = 1e5;         % 采样频率 (Hz)
window_length = 2048; % STFT 窗口长度 (样本点数)
overlap = 1024;   % 窗口重叠 (样本点数)
nfft = 2048;      % FFT 点数
threshold = -20;  % 分离信号的阈值 (dB)

% 降采样比例，控制时间维度的分辨率
downsample_factor = 32; % 每隔 8 个时间点取一个样本进行降采样

% 根目录路径设置
input_high_root = 'F:/ccc/DroneData/AllHigh'; % 高频信号根目录
input_low_root = 'F:/ccc/DroneData/AllLow';   % 低频信号根目录
output_root = 'F:/ccc/DroneData3';    % 保存拼接后的信号根目录

% 获取根目录下的所有子文件夹
subfolders_high = dir(input_high_root);
subfolders_low = dir(input_low_root);

% 只保留文件夹
subfolders_high = subfolders_high([subfolders_high.isdir]);
subfolders_low = subfolders_low([subfolders_low.isdir]);

% 处理每个子文件夹
for subfolder_idx = 1:length(subfolders_high)
    % 获取当前子文件夹的路径
    high_folder = fullfile(input_high_root, subfolders_high(subfolder_idx).name);
    low_folder = fullfile(input_low_root, subfolders_low(subfolder_idx).name);

    % 确保当前子文件夹路径下的低频信号文件夹存在
    if ~isfolder(low_folder)
        warning('低频信号文件夹 %s 不存在', low_folder);
        continue; % 跳过当前子文件夹
    end

    % 为当前子文件夹创建输出文件夹
    output_folder = fullfile(output_root, subfolders_high(subfolder_idx).name);
    if ~isfolder(output_folder)
        mkdir(output_folder); % 如果输出文件夹不存在，则创建
    end

    % 获取当前子文件夹下的所有高频和低频文件
    csv_high_files = dir(fullfile(high_folder, '*.csv'));
    csv_low_files = dir(fullfile(low_folder, '*.csv'));

    % 初始化合并的数据结构
    all_stft_combined = [];
    all_f_combined = [];
    all_magnitude_normalized = [];

    % 处理每个对应的文件
    for file_idx = 1:length(csv_high_files)
        % 获取对应的低频文件名
        high_file_name = csv_high_files(file_idx).name;
        low_file_name = strrep(high_file_name, 'H', 'L'); % 替换 'H' 为 'L' 获取低频文件名

        % 检查对应的低频文件是否存在
        low_file_path = fullfile(low_folder, low_file_name);
        if isfile(low_file_path)
            % 加载高频时域信号
            high_signal = csvread(fullfile(high_folder, high_file_name))'; % 转置为行向量
            [stft_high, f_high, t_high] = spectrogram(high_signal, hamming(window_length), overlap, nfft, fs);

            % 加载低频时域信号
            low_signal = csvread(low_file_path)'; % 转置为行向量
            [stft_low, f_low, t_low] = spectrogram(low_signal, hamming(window_length), overlap, nfft, fs);

            % 拼接 STFT 矩阵
            stft_combined = [stft_low; stft_high]; % 在频率维度上拼接
            f_combined = [f_low; f_high]; % 合并频率信息

            % 计算幅度谱并归一化
            magnitude_combined = abs(stft_combined); % 计算幅度谱
            magnitude_normalized = (magnitude_combined - min(magnitude_combined(:))) / ...
                                   (max(magnitude_combined(:)) - min(magnitude_combined(:))); % 归一化到 [0, 1]

            % ===============================
            % 时间维度降采样
            % ===============================
            % 合并时间轴
            % 时间轴是一样的,所以不及进行合并

            t_combined = t_low;

            % 降采样时，确保 time_indices 不超出列数
            time_indices = 1:downsample_factor:min(length(t_combined), length(t_combined)); % 限制索引范围

            % 降采样 STFT
            stft_combined_downsampled = stft_combined(:, time_indices); % 降采样时频矩阵
            magnitude_normalized_downsampled = magnitude_normalized(:, time_indices); % 降采样后的幅度谱

            % 更新合并的数据
            all_stft_combined = [all_stft_combined, stft_combined_downsampled];
            all_f_combined = [all_f_combined; f_combined]; % 合并频率信息
            all_magnitude_normalized = [all_magnitude_normalized, magnitude_normalized_downsampled];
        else
            warning('对应的低频文件 %s 不存在', low_file_name);
        end
    end

    % 每处理完一个文件夹，将合并后的数据保存为 MAT 文件
    final_combined_file = fullfile(output_folder, 'all_combined_data.mat');
    save(final_combined_file, 'all_stft_combined', 'all_f_combined', 'all_magnitude_normalized', '-v7.3');
end
