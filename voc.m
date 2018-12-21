startup;
% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('imagenet-vgg-m')) ;
% net = vl_simplenn_tidy(load('imagenet-vgg-m')) ;
gpuDevice(1);
net.move('gpu');
% vl_simplenn_move(net, 'gpu');
net.mode = 'test';
m = 50;
dirPre = '/home/data/ML_Data/';
class_label = {'person', 'bird','cat','cow','dog','horse','sheep', 'aeroplane','bicycle','boat','bus','car','motorbike','train', 'bottle','chair','diningtable','pottedplant','sofa','tvmonitor'};
imgs_path = '/home/data/VOC2012/JPEGImages/'; 
ann_path = '/home/data/VOC2012/ImageSets/Main/';
set = {'train', 'test'};
txt = {'train', 'val'};
for s=1:2
    indx = 1;
    for i=1:numel(class_label)
        data_feature_all = [];
        labels_0_all = [];
        labels_1_all = [];
        txtFile = fopen([ann_path, class_label{i}, '_', txt{s}, '.txt']);
        lines = textscan(txtFile, '%s%d');
        % load and preprocess an image
        coarse = 1;
        if i==1
            coarse = 1;
        elseif (i>=2&&i<=7)
            coarse = 2;
        elseif (i>=8&&i<=14)
            coarse = 3;
        else
            coarse = 4;
        end
        for j=1:numel(lines{2})
            if lines{2}(j) == 1
                im_ = imread([imgs_path, lines{1}{i}, '.jpg']);
                if size(im_, 3) == 3
                   im_ = single(im_);
                   im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
                   im_ = im_ - net.meta.normalization.averageImage ;
                else
                   im_ = single(repmat(img,[1 1 3])) ; % note: 255 range
                   im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
                   im_ = im_ - net.meta.normalization.averageImage ;
                end
                im_gpu = gpuArray(im_);
                net.conserveMemory = 0;
                net.eval({'x0', im_gpu});
                feat = net.vars(15).value;
                feat = reshape(gather(feat), 169, 512);
                data_feature_all{end+1} = feat;
                labels_0_all(end+1) = i;
                labels_1_all(end+1) = coarse;
                indx = indx + 1;
                fprintf('processing %d-th image...\n', indx);
            end
        end
        fclose(txtFile);
        [~, inds]=datasample([1:numel(labels_0_all)], m, 'Replace', false);
        data_feature = cell(1,m);
        labels_0 = zeros(1,m);
        labels_1 = zeros(1,m);
        for p=1:m
            data_feature{p} = data_feature_all{inds(p)};
            labels_0(p) = labels_0_all(inds(p));
            labels_1(p) = labels_1_all(inds(p));
        end
        save(sprintf([dirPre, '/pascalvoc/', set{s},'/pascalvoc_',  set{s}, '_%d_%d.mat'], i, coarse), 'data_feature', 'labels_0', 'labels_1');
    end
end
