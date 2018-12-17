function [tdt, trt, fRate]=rmml_rie(dataset, lam, tt)

tic
S = generatePosSamples(dataset);
D = generateNegSamples(dataset);
tdt=toc;
params=[lam;tt];
Q_0 = eye(513);

% training code 

[trt,Q_rmml] = rmml_train(S, D, Q_0, params);
 
% %% calculate similarity
% 
% sim_mat = zeros(length(labeltrain),length(labeltest));
% for i = 1 : length(labeltrain)
%     T1=LogC(:,:,i);
%     for j = 1 : length(labeltest)
%         T2=LogC_t(:,:,j);
%         T = T1-T2;
%         T = T'*T;
%         sim_mat(i,j) = trace(Q_rmml*T);
%     end
% end
% 
% %% calculate accuracy
% 
% sampleNum = length(labeltest);
% [~, ind] = sort(sim_mat,1,'ascend');
% correctNum = length(find((labeltest-labeltrain(ind(1,:)))==0));
% fRate = correctNum/sampleNum;
% fprintf('fRate = %f \n', fRate);

end
