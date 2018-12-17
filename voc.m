startup;
% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('imagenet-vgg-m')) ;
% net = vl_simplenn_tidy(load('imagenet-vgg-m')) ;
gpuDevice(1);
net.move('gpu');
% vl_simplenn_move(net, 'gpu');
net.mode = 'test';
class_label = {'person', 'bird','cat','cow','dog','horse','sheep', 'aeroplane','bicycle','boat','bus','car','motorbike','train', 'bottle','chair','diningtable','pottedplant','sofa','tvmonitor'};
imgs_path = '/home/data/VOC2012/JPEGImages/'; 
ann_path = '/home/data/VOC2012/ImageSets/Main/';
indx = 1;
for i=1:numel(class_label)
    data_feature = [];
    labels_0 = [];
    labels_1 = [];
    txtFile = fopen([ann_path, class_label{i}, '_train.txt']);
    lines = textscan(txtFile, '%s%d');
    % load and preprocess an image
    coarse = 0;
    if i==1
        coarse = 0;
    elseif (i>=2&&i<=7)
        coarse = 1;
    elseif (i>=8&&i<=14)
        coarse = 2;
    else
        coarse = 3;
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
            data_feature{end+1} = feat;
            labels_0{end+1} = i-1;
            labels_1{end+1} = coarse;
            indx = indx + 1;
            fprintf('processing %d-th image...\n', indx);
        end
    end
    fclose(txtFile);
    save(sprintf('./feature/pascalvoc/train/pascalvoc_train_%d_%d.mat', i-1, coarse), 'data_feature', 'labels_0', 'labels_1');
end
