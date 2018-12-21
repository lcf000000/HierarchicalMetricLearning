function D = generateNegSamples(dataset, m, dirPre)
%GENERATENEGSAMPLES Summary of this function goes here
%   Detailed explanation goes here
    len = load([dirPre, dataset, '/', dataset, '_train_length.mat' ]);
    file_list = dir([dirPre, dataset, '/spdFeature_train/', dataset, '_train_*.mat' ]);
    class_num = length(file_list);
    D = zeros(513, 513);
    
    % generate mark struct
    picked = [];
    for i=1:len.superClass_num
        for j=1:length(len.sub_len{i})
            for p=1:length(len.sub_len{i})
                if len.sub_len{i}{p} ~= len.sub_len{i}{j}
                    oClass = [];
                    picked{i, len.sub_len{i}{j}, len.sub_len{i}{p}} = oClass;
                end
            end
        end
    end
    
    for i=1:class_num
        file = dir([dirPre, dataset, '/spdFeature_train/', dataset, '_train_', num2str(i), '_*.mat']);
        select_class = load([dirPre, dataset, '/spdFeature_train/', file.name]);
        super_class = select_class.labels_1(1);
        sub_list = dir([dirPre, dataset, '/spdFeature_train/', dataset, '_train_*_', num2str(super_class), '.mat']);
        subClass_num = length(sub_list);
        anchor_num = numel(select_class.labels_0);
        for j=1:anchor_num
            anchor = anchors.data_feature(:, :, j);
            for p=1:subClass_num
                if ~strcmp(sub_list(p).name, file.name)
                    sample = load([dirPre, dataset, '/spdFeature_train/', sub_list(p).name]);
                    selected_len = 0;
                    while(selected_len >= m)
                        ind = minDistance(anchor, sample.data_feature, true, m);
                        for q=1:m
                            if selected_len >= m
                                break;
                            end
                            [select picked] = chooseSamples(picked, super_class, anchor.labels_0(1), sample.labels_0(1), j, ind(q));
                            if select
                                selected_len = selected_len +1;
                                xx = anchor - sample.data_feature(:, :, ind(q));
                                D = D + xx'*xx;
                            end
                        end
                    end
                    clear sample;
                end
            end
        end
        clear select_class;
    end
end

function [selected picked] =  chooseSamples(picked, super_class, anchor_class, sample_class, anchor_ind, sample_ind)
    selected = false;
    if (sample_class <= size(picked, 2) || anchor_class <= size(picked, 3))
        pick= picked{super_class, sample_class, anchor_class};
        if lenght(find(pick == anchor_ind))==0
           picked{super_class, sample_class, anchor_class}(end+1) = sample_ind;
           selected = true;
        end
    else
        picked{super_class, sample_class, anchor_class}(end+1) = sample_ind;
        selected = true;
    end
end
    

