function S = generatePosSamples(dataset, m)
%GENERATESAMPLE Summary of this function goes here
%   Detailed explanation goes here
    dirPre = '/home/data/ML_Data/';
    file_list = dir([dirPre, dataset, '/spdFeature_train/', dataset, '_train_*.mat' ]);
    class_num = length(file_list);
    S = zeros(513, 513);
    for i=1:class_num
        load([dirPre, dataset, '/spdFeature_train/', file_list(i).name]);
        anchor_num = numel(labels_0);
        picked = zeros(anchor_num, anchor_num);
        for j=1:anchor_num
            anchor = data_feature(:, :, j);
            selected = minDistance(anchor, data_feature, false, m);
            for p=1:m
                if picked(p,j)~=1;
                    picked(j,p) = 1;
                    xx = anchor - data_feature(:, :, selected(p));
                    S = S + xx'*xx;
                end
            end
        end
    end
end

