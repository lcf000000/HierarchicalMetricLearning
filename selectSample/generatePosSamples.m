function S = generatePosSamples(dataset, m)
%GENERATESAMPLE Summary of this function goes here
%   Detailed explanation goes here
    file_list = dir(['../feature/', dataset, '/spdFeature_train/', dataset, '_train_*.mat' ]);
    class_num = length(file_list);
    S = zeros(513, 513);
    for i=1:class_num
        file = dir(['../feature/', dataset, '/spdFeature_train/', dataset, '_train_', num2str(i), '_*.mat']);
        load(['../feature/', dataset, '/spdFeature_train/', file.name]);
        anchor_num = length(data_feature);
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

