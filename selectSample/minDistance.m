function selected = minDistance(anchor, data_feature, pair_flag, k)
%SELECTSAMPLE Summary of this function goes here
%   Detailed explanation goes here
	len = size(data_feature, 3);
    distance = [];
    for i=1:len
        if(pair_flag)
            sample = data_feature(:, :, i);
            distance(end+1) = trace((anchor-sample)*(anchor-sample));
        else
            if(i~=anchor)
                sample = data_feature(:, :, i);
                distance(end+1) = trace((anchor-sample)*(anchor-sample));
            end
        end
    end
    [~, selected] = sort(distance);
    selected = selected(1:k);
end

