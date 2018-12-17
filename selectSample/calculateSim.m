function [ output_args ] = calculateSim(dataset)
    %CALCULATESIM Summary of this function goes here
    %   Detailed explanation goes here
    train_len = load(['../feature/', dataset, '/cifar100_train_length.mat' ]);
    test_len = load(['../feature/', dataset, '/cifar100_test_length.mat' ]);
    train_list = dir(['../feature/', dataset, '/spdFeature_train/', dataset, '_train_*.mat' ]);
    test_list = dir(['../feature/', dataset, '/spdFeature_test/', dataset, '_test_*.mat' ]);
    sim_mat = zeros(train_len, test_len);
    test_num = length(test_list);
    for i=1:test_num
        test = load(['../feature/', dataset, '/spdFeature_test/', test_list(i).name]);
        for j=1:length(test.labels_0)
            T1 = test.data_feature(:,:,j);
            
            for p=1:test_num
                T2=LogC_t(:,:,j);
                T = T1-T2;
                T = T'*T;
                sim_mat(i,j) = trace(Q_rmml*T);
            end
        end
end
