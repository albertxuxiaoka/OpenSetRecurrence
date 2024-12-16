

% 执行TE特征提取
traverse_mat_files('F:/ccc/DroneData3');

function traverse_mat_files(directory)
    % 指定保存的文件夹路径
    folder_path = "F:/ccc/DroneFeature/TE_feature";  % 你可以将这个路径修改为你自己的文件夹路径

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
            
            %获取当前数据所属文件夹
            [~, folderName, ~] = fileparts(directory);
            file_name = folderName + ".mat";

            
            % 在这里可以进行对 .mat 文件的处理
            stftData = loaded_data.all_magnitude_normalized;
            texture_features = [];
            % 每次遍历的列数
            col = size(stftData,2); 
            cols_per_iteration = ceil(col / 10);

            % 进行10次遍历
            for j = 1:10  % 将 i 改为 j
                % 计算当前遍历的列的起始和结束索引
                start_col = (j - 1) * cols_per_iteration + 1;  % 将 i 改为 j
                end_col = min(j * cols_per_iteration, col);      % 将 i 改为 j

                % 提取当前遍历的部分矩阵（按列）
                sub_matrix = stftData(:, start_col:end_col);

                % 根据 j 进行某些操作，例如将子矩阵的元素乘以当前的迭代次数 j
                temp_feature = texture_extractor(sub_matrix);
                texture_features = [texture_features ; temp_feature];

                % 打印每次遍历的信息
                disp(['Iteration ', num2str(j), ': processed columns ', num2str(start_col), ' to ', num2str(end_col)]);

                % 这里可以继续对 sub_matrix 执行其他操作
            end
            
            
            % 对数据添加标签并且保存
            %labeled_features = [texture_features, i];
            texture_file = fullfile(folder_path, file_name);
            save(texture_file, 'texture_features');     
            
        end
    end
end
