lamda=0.1;
t=0.6;
m=100;
dataset = 'cifar100';
[tdt, trd, Q]=rmml_rie(dataset, lamda, t, m);
save(['../feature/', dataset, '/', dataset, '_Q.mat' ], Q);
fprintf('tdt_time = %.2f \n', tdt/3600);
fprintf('trd_time = %.2f \n', trd/3600);