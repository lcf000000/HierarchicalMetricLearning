function CY1 = Compute_Log_Cov(SY1)
%calculate logarithm of SPD matrix

number_sets1=length(SY1);
for tmpC1=1:number_sets1
    Y1=SY1{tmpC1};
    %Y1 = transpose(Y1);
    y1_mu = mean(Y1,2);       
    
    Y1 = Y1-repmat(y1_mu,1,size(Y1,2));
    Y1 = Y1*Y1'/(size(Y1,2)-1);
    lamda = 0.001*trace(Y1);
    Y1 = Y1+lamda*eye(size(Y1,1));   
    
%     a=max(det(Y1),1e-12);
%     Y1 = a^(-1/(size(Y1,1)+1))*[Y1+y1_mu*y1_mu' y1_mu;y1_mu' 1];
    Y1=[Y1+y1_mu*y1_mu' y1_mu;y1_mu' 1];
    CY1(:,:,tmpC1)= logm(Y1);
end