function [labels_train, labels_test, sim_mat] = calculateSim(Q_rmml, dataset)
    %CALCULATESIM Summary of this function goes here
    %   Detailed explanation goes here
    dirPre = '/home/data/ML_Data/';
    test_len = load([dirPre, dataset, '/cifar100_test_length.mat' ]);
    train_len = load([dirPre, dataset, '/cifar100_train_length.mat' ]);
    test_list = dir([dirPre, dataset, '/spdFeature_test/', dataset, '_test_*.mat' ]);
    train_list = dir([dirPre, dataset, '/spdFeature_train/', dataset, '_train_*.mat' ]);
    sim_mat = zeros(train_len.sum, test_len.sum);
    labels_train = zeros(train_len.sum);
    labels_test = zeros(test_len.sum);
    test_num = length(test_list);
    train_num = length(train_list);
    idx_train = 1;
    for i=1:train_num
        train = load([dirPre, dataset, '/spdFeature_train/', train_list(i).name]);
        for p=1:length(train.labels_0)
            T1 = train.data_feature(:,:,p);
            labels_train(1, idx_train) = train.labels_0(1);
            idx_test = 1;
            for j=1:test_num
                test = load([dirPre, dataset, '/spdFeature_test/', test_list(i).name]);
                for q=1:length(test.labels_0)
                    T2 = test.data_feature(:,:,q);
                    labels_test(1, idx_test) = test.labels_0(1);
                    T = T1-T2;
                    T = T'*T;
                    sim_mat(idx_train, idx_test) = trace(Q_rmml*T);
                    idx_train = idx_train + 1;
                    idx_test = idx_test + 1;
                end
            end
        end
    end
end

