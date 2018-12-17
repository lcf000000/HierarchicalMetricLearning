function spdFeature(dataset, set)
%SPDCACHE Summary of this function goes here
%   Detailed explanation goes here
    file_list = dir(['../feature/', dataset, '/',  set, '/', dataset, '_', set, '_*.mat' ]);
    class_num = length(file_list);
    for i=1:class_num
        load(['../feature/', dataset, '/',  set, '/', file_list(i).name]);
        data_feature = Compute_Log_Cov(data_feature);
        fprintf('%s...\n',file_list(i).name);
        save(['../feature/', dataset, '/spdFeature_',  set, '/', file_list(i).name], 'data_feature', 'labels_0', 'labels_1');
    end
end

