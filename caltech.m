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
data_path = '/home/data/256_ObjectCategories/';
dir_list = dir(data_path);
food = {'beer-mug','coffe-mug','ewer-101','soda-can','win-bottle','teapot', 'chopsticks','straw','spoon','frying-pan', 'cake','ice-cream-cone','fried-egg','hamburger','hot-dog','spaghetti','sushi'};
set = {'train', 'test'};
indx = 1;
class_num = 0;
for i=3:length(dir_list)
    dirname = strsplit(dir_list(i).name, '.');
    if ~isempty(find(strcmp(food, dirname{2})))
        class_num = class_num + 1;
        data_feature_all = [];
        labels_0_all = [];
        labels_1_all = [];
        % load and preprocess an image
        coarse = 1;
        if (class_num>=1&&class_num<=6)
            coarse = 1;
        elseif (class_num>=7&&class_num<=10)
            coarse = 2;
        else
            coarse = 3;
        end
        img_list = dir([data_path, dir_list(i).name]);
        for j=3:length(img_list)
            filename = strsplit(img_list(j).name, '.');
            if strcmp(filename(2),'jpg')
                im_ = imread([data_path, dir_list(i).name, '/', filename(2), '.jpg']);
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
                labels_0_all(end+1) = class_num;
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
