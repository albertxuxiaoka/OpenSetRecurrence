input_file = "F:/ccc/DroneFeature/PE_feature_T"
output_file = "F:\ccc\DroneFeature\round2_feature";
files = dir(input_file)
time_feature = [];
for i = 1: length(files)
    %排除特殊文件
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
            continue;
    end
    %读取数据
    full_path = fullfile(input_file, files(i).name);
    temp_data = load(full_path);
    %添加标签
    %label = (i - 2) * ones(10,1);
    %temp_data = [temp_data, label];
    time_feature = [time_feature;temp_data.T_features_combined];
    
end

file_name = "time_feature.mat";
save(fullfile(output_file,file_name),'time_feature');
    