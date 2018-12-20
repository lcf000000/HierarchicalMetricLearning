function spdFeature(dataset, dirPre)
%SPDCACHE Summary of this function goes here
%   Detailed explanation goes here
    set = {'train', 'test'};
    for s=1:2
        file_list = dir([dirPre, dataset, '/',  set{s}, '/', dataset, '_', set{s}, '_*.mat' ]);
        class_num = length(file_list);
        for i=1:class_num
            load([dirPre, dataset, '/',  set{s}, '/', file_list(i).name]);
            data_feature = Compute_Log_Cov(data_feature);
            fprintf('%s...\n',file_list(i).name);
            save([dirPre, dataset, '/spdFeature_',  set{s}, '/', file_list(i).name], 'data_feature', 'labels_0', 'labels_1');
        end
    end
end

