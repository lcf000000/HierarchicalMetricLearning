function [tdt, trt, fRate]=rmml_rie(dataset, lam, tt, m, dirPre)
% tic
% S = generatePosSamples(dataset, m, dirPre);
% save([dirPre, dataset, '/', dataset,'_S.mat'], 'S');
% fprintf('Positive samples done ! \n');
% D = generateNegSamples(dataset, m, dirPre);
% save([dirPre, dataset, '/', dataset,'_D.mat'], 'D');
% fprintf('Negtive samples done ! \n');
% tdt=toc;

tdt = 9600;
load([dirPre, dataset, '/', dataset, '_S.mat']);
load([dirPre, dataset, '/', dataset, '_D.mat']);

params=[lam;tt];
Q_0 = eye(513);

% training code 

[trt, Q_rmml] = rmml_train(S, D, Q_0, params);
 
% calculate similarity

[labels_train, labels_test, sim_mat] = calculateSim(Q_rmml, dataset, dirPre);
save([dirPre, dataset, '/', dataset,'_sim.mat'], 'sim_mat');
% calculate accuracy

sampleNum = size(sim_mat, 2);
[~, ind] = sort(sim_mat,1,'ascend');
correctNum = length(find((labels_test-labels_train(ind(1,:)))==0));
fRate = correctNum/sampleNum;
fprintf('fRate = %f \n', fRate);

end
