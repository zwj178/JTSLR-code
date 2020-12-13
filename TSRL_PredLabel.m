function Yt = TSRL_PredLabel(P,Q,X_tar_test)
% solve the optimization problem of Eq.(7) in our SPL paper
% inputs:
%       P Q -- the optimal P Q learned by GTLSR.m
%       Yt1 -- d*Nt target feature matrix, where d is the dimension of the
%       feature vector and Nt is the sample number.
A1=P*Q';
d  = size(A1,1);
I = eye(d);
Aeq = ones(1,d);
b = zeros(d,1);
bx = zeros(d,1);
A = -I;
options=optimset('Algorithm','interior-point-convex');

for i=1:size(X_tar_test,2)
%     tmpxt =(A1*Yt1(:,i)+A2*Yt2(:,i)+A3*Yt3(:,i))/3;
    tmpxt =A1*X_tar_test(:,i);
    f = -2*tmpxt;
    Yt(:,i) = quadprog(I,f,A,b,Aeq,1,zeros(d,1),ones(d,1),bx,options);
end
