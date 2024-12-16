% 执行TE特征提取
traverse_mat_files('F:/ccc/DroneData3');

function traverse_mat_files(directory)
    % 指定保存的文件夹路径
    folder_path_T = "F:/ccc/DroneFeature/PE_feature_T";  % 你可以将这个路径修改为你自己的文件夹路径
    folder_path_F = "F:/ccc/DroneFeature/PE_feature_F";  % 你可以将这个路径修改为你自己的文件夹路径

    % 遍历指定目录及其子目录中的所有 .mat 文件
    % directory: 要遍历的根目录路径
    
    % 获取目录下的所有文件和文件夹
    files = dir(directory);
    
    % 遍历每个文件/文件夹
    for i = 1:length(files)
        % 排除 '.' 和 '..' 目录
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
            continue;
        end
        
        % 构建文件或文件夹的完整路径
        full_path = fullfile(directory, files(i).name);
        
        % 如果是文件夹，则递归遍历
        if files(i).isdir
            fprintf('进入子文件夹: %s\n', full_path);
            traverse_mat_files(full_path);  % 递归调用遍历子文件夹
        elseif endsWith(files(i).name, '.mat')
            % 如果是 .mat 文件，执行相应的操作
            fprintf('找到 .mat 文件: %s\n', full_path);
            % 例如，加载文件：
            loaded_data = load(full_path);
            
            % 获取当前数据所属文件夹
            [~, folderName, ~] = fileparts(directory);
            file_name = folderName + ".mat";
            
            % 在这里可以进行对 .mat 文件的处理
            stftData = loaded_data.all_magnitude_normalized;

            % 初始化用于保存10部分特征的数组
            T_features_combined = [];
            F_features_combined = [];
            
            % 数据划分为10份
            col = size(stftData, 2); 
            cols_per_iteration = ceil(col / 10);

            % 分10份处理
            for j = 1:10
                % 当前分块的列范围
                start_col = (j - 1) * cols_per_iteration + 1;
                end_col = min(j * cols_per_iteration, col);
                
                % 提取当前部分数据
                sub_matrix = stftData(:, start_col:end_col);

                % 处理当前部分数据并提取特征
                %sub_matrix_sampled = Sample_Extractor(sub_matrix);
                [T_feature, F_feature] = PE_seperate(sub_matrix);

                % 保存当前部分特征
                T_features_combined = [T_features_combined; T_feature'];
                F_features_combined = [F_features_combined; F_feature'];

                % 打印进度信息
                disp(['部分 ', num2str(j), ': 处理列 ', num2str(start_col), ' 到 ', num2str(end_col)]);
            end
            
            % 保存合并后的时间和频率特征
            position_file_T = fullfile(folder_path_T, file_name);
            save(position_file_T, 'T_features_combined');
            position_file_F = fullfile(folder_path_F, file_name);
            save(position_file_F, 'F_features_combined');
        end
    end
end
