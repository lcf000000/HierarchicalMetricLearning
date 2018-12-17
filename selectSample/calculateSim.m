function sim_mat = calculateSim(Q_rmml, dataset)
    %CALCULATESIM Summary of this function goes here
    %   Detailed explanation goes here
    test_len = load(['../feature/', dataset, '/cifar100_test_length.mat' ]);
    train = load(['../feature/', dataset, '/', dataset, '_sim_train.mat' ]);
    test_list = dir(['../feature/', dataset, '/spdFeature_test/', dataset, '_test_*.mat' ]);
    train_len = length(train.labels_0);
    sim_mat = zeros(train_len, test_len.sum);
    test_num = length(test_list);
    idx_test = 1;
    for i=1:test_num
        test = load(['../feature/', dataset, '/spdFeature_test/', test_list(i).name]);
        for j=1:length(test.labels_0)
            T1 = test.data_feature(:,:,j);
            for p=1:train_len
                T2 = train.data_feature(:,:,p);
                T = T1-T2;
                T = T'*T;
                sim_mat(idx_test, p) = trace(Q_rmml*T);
            end
        end
    end
end

