function generateSimTrain(dataset, k)
%GENERATESIMTRAIN Summary of this function goes here
%   Detailed explanation goes here
    train_list = dir(['../feature/', dataset, '/spdFeature_train/', dataset, '_train_*.mat' ]);
    train_num = length(train_list);
    data_feature = zeros(513, 513, k*train_num);
    labels_0 = cell(1, k*train_num);
    labels_1 = cell(1, k*train_num);
    for i=1:train_num
        train = load(['../feature/', dataset, '/spdFeature_train/', train_list(i).name]);
        [~, inds]=datasample(train.labels_0, k);
        for j=1:numel(inds)
            idx = (i-1)*k+j;
            data_feature(:,:,idx);
            labels_0{idx} = train.labels_0{j};
            labels_1{idx} = train.labels_1{j};
        end
    end
    save(['../feature/', dataset, '/', dataset, '_sim_train.mat' ], '-v7.3', 'data_feature', 'labels_0', 'labels_1');
end

