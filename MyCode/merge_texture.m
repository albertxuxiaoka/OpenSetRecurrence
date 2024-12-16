input_file = "F:/ccc/DroneFeature/TE_feature"
output_file = "F:\ccc\DroneFeature\round2_feature";
files = dir(input_file)
texture_feature = [];
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
    texture_feature = [texture_feature;temp_data.texture_features];
    
end

file_name = "texture_feature.mat";
save(fullfile(output_file,file_name),'texture_feature');
    