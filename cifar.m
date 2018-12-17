startup;
% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('imagenet-vgg-m')) ;
% net = vl_simplenn_tidy(load('imagenet-vgg-m')) ;
gpuDevice(4);
net.move('gpu');
% vl_simplenn_move(net, 'gpu');
net.mode = 'test';

% class_label = ['',''];
% imgs_path = '/home/data/VOC2012/JPEGImages/'; 
% ann_path = '/home/data/VOC2012/ImageSets/Main/';

% load and preprocess an image
load('./cifar-100-matlab/train.mat');
for j=1:numel(unique(fine_labels))
    data_feature = [];
    labels_0 = [];
    labels_1 = [];
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
            feat = reshape(gather(feat), 169, 512);
            data_feature{end+1} = feat;
            labels_0{end+1} = fine_labels(i);
            labels_1{end+1} = coarse_labels(i);
            fprintf('processing %d-th image...\n', i);
        end
    end
    save(sprintf('./feature/cifar100/train/cifar100_train_%d_%d.mat', j-1, labels_1{1}), 'data_feature', 'labels_0', 'labels_1');
end