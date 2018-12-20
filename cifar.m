startup;
% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('imagenet-vgg-m')) ;
% net = vl_simplenn_tidy(load('imagenet-vgg-m')) ;
gpuDevice(4);
net.move('gpu');
% vl_simplenn_move(net, 'gpu');
net.mode = 'test';
m = 50;

% class_label = ['',''];
% imgs_path = '/home/data/VOC2012/JPEGImages/'; 
% ann_path = '/home/data/VOC2012/ImageSets/Main/';

% load and preprocess an image
set = {'train', 'test'};
for s=1:2
    load(['./cifar-100-matlab/', set{s},'.mat']);
    for j=1:numel(unique(fine_labels))
        data_feature_all = [];
        labels_0_all = [];
        labels_1_all = [];
        for i=1:numel(fine_labels)
            if fine_labels(i)==j-1
                im_ = reshape(data(i,:), 32, 32, 3);
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
                feat = reshape(gather(feat), 512, 169);
                data_feature_all{end+1} = feat;
                labels_0_all(end+1) = fine_labels(i)+1;
                labels_1_all(end+1) = coarse_labels(i)+1;
                fprintf('processing %d-th image...\n', i);
            end
        end
        [~, inds]=datasample([1:numel(labels_0_all)], m);
        data_feature = cell(1,m);
        labels_0 = zeros(1,m);
        labels_1 = zeros(1,m);
        for p=1:m
            data_feature{p} = data_feature_all{inds(p)};
            labels_0(p) = labels_0_all(inds(p));
            labels_1(p) = labels_1_all(inds(p));
        end
        save(sprintf(['./feature/cifar100/', set{s},'/cifar100_', set{s},'_%d_%d.mat'], j, labels_1_all(1)), 'data_feature', 'labels_0', 'labels_1');
    end
end