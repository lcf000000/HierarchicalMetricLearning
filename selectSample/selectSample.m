function pos_sample = selectSample(anchor, data_feature)
%SELECTSAMPLE Summary of this function goes here
%   Detailed explanation goes here
	len = length(data_feature);
    distance = ones(len, 1);
    anchor = Compute_Log_Cov(anchor);
    for i=1:len
        sample = Compute_Log_Cov(data_feature(i));
        distance(end+1) = trace((anchor-sample)*(anchor-sample));
    end
    [~, pos_sample] = min(distance);
end

