lamda=0.1;
t=0.6;
m=25;
dataset = 'cifar100';
dirPre = '/home/data/ML_Data/';
name = [];
Rate = [];
time = [];
[tdt, trd, fRate]=rmml_rie(dataset, lamda, t, m, dirPre);
name{end+1} = 'Ours';
Rate(end+1) = fRate;
time(end+1) = trd;
save([dirPre, dataset, '/', dataset, '_Rate.mat' ], 'name', 'Rate', 'time');
fprintf('tdt_time = %.2f \n', tdt/3600);
fprintf('trd_time = %.2f \n', trd/3600);
