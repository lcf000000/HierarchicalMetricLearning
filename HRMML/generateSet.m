function generateSet(dataset, dirPre)
%GENERATESET Summary of this function goes here
%   Detailed explanation goes here
    set = {'train', 'test'};
    for s=1:2
        sum = 0;
        super_class = [];
        file_list = dir([dirPre, dataset, '/spdFeature_',  set{s}, '/', dataset, '_', set{s}, '_*.mat' ]);
        class_num = length(file_list);
        sub_len = [];
        for i=1:class_num
            load([dirPre, dataset, '/',  set{s}, '/', file_list(i).name]);
            super_class(end+1) = labels_1(1);
        end
        superClass_num = numel(unique(super_class));
        for i=1:superClass_num
            sub_len{i} = [];
        end
        for i=1:class_num
            load([dirPre, dataset, '/',  set{s}, '/', file_list(i).name]);
            sub_len_ = length(labels_0);
            sub_len{labels_1(1)}{end+1} = labels_0(1);
            sum = sum + sub_len_;
        end
        save([dirPre, dataset, '/', dataset, '_', set{s}, '_length.mat' ], 'sum','superClass_num','sub_len');
    end
end

