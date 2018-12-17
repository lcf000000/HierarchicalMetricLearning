function [t, Q] = rmml_train(S, D, Q_0, params )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%  S=S/sum_s;
%  D=D/sum_d;
tic
%lamda=0.1 or 0  t=0:0.1:1 
lamda=params(1);
t=params(2);
M=S+lamda*inv(Q_0);        
N=D+lamda*Q_0;
Q=expm((-(1-t)*logm(M)+t*logm(N))/2);
% M=inv(S+lamda*inv(Q_0));        
% N=D+lamda*Q_0;
% Q=M^(1/2)*(M^(-1/2)*N*M^(-1/2))^t*M^(1/2);  %利用式子直接求出Q
t=toc;
% M=inv(S+lamda*inv(Q_0));        
% N=D+lamda*Q_0;
% Q = real(sharp(inv(M),N,t));

end

