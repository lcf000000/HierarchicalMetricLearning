function generateSimSet(dataset, set)
%GENERATESIMTRAIN Summary of this function goes here
%   Detailed explanation goes here
    k = 50;
 dirPre = '/home/data/ML_Data/';
    file_list = dir([dirPre, dataset, '/spdFeature_', set, '/', dataset, '_', set, '_*.mat' ]);
    train_num = length(file_list);
    data_feature = zeros(513, 513, k*train_num);
    labels_0 = zeros(1, k*train_num);
    labels_1 = zeros(1, k*train_num);
    for i=1:train_num
        class = load([dirPre, dataset, '/spdFeature_', set, '/', file_list(i).name]);
        data_feature(:,:,(i-1)*k+1:i*k) = class.data_feature;
        labels_0((i-1)*k+1:i*k) = class.labels_0;
        labels_1((i-1)*k+1:i*k) = class.labels_1;
    end
    save([dirPre, dataset, '/', dataset, '_sim_', set, '.mat' ], '-v7.3', 'data_feature', 'labels_0', 'labels_1');
end

