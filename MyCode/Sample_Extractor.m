function TFS_sampled = Sample_Extractor(TFS)
% 假设 TFS 是一个时频图，大小为 M×N（M 为频率点数，N 为时间窗口数）
% M = 1024，N = 9764 作为示例

[M, N] = size(TFS);  % 获取时频图的尺寸
num_time_points = floor(N / 2);  % 取时间轴上一半的点

% 1. 生成均匀采样的时间点下标
selected_time_indices = round(linspace(1, N, num_time_points));

% 2. 对时频图进行切片，取采样点
TFS_sampled = TFS(:, selected_time_indices);

% 输出采样后的时频图
%disp('采样后的时频图尺寸：');
%disp(size(TFS_sampled));  % 应为 [M, num_time_points]
end
