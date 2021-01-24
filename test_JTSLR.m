
X_s=Atrain_matrix;
X_s = normalization(X_s',1);
Y_s_label=Atrain_label;
X_tr =Btrain_matrix;  % the auxiliary set
X_tr = normalization(X_tr',1);
X_te = Btest_matrix;%target
X_te = normalization(X_te',1);  
Y_tr_label = Btrain_label;
Y_te_label = Btest_label;

%l=[1 5 10 50 100 500 1000];
%a=[0.001 0.01 0.1 1 10 100 1000];
%for j=1:7
options.lambda=100; 
options.gamma=0;
options.dim=100;
options.alpha=0;    
options.T=10;
[P,Q] = JTSLR(Y_s_label,X_s,X_tr,Y_tr_label,options);

A_new=P*Q'*X_s;
B_new=P*Q'*X_te;
%model= libsvmtrain(Atrain_label,A_new','-s 0 -t 0 -c 1 -g 1 ');
%[predict_label, accuracy, dec_values] = libsvmpredict(Btest_label,B_new', model);

%a(:,j)=accuracy(1);
pred_Y_te = TSRL_PredLabel(P,Q,X_te);
[Max_value,te_label] = max(pred_Y_te);

n=0;
for i=1:length(te_label)
    if te_label(i)==Y_te_label(i)
        n=n+1;
    end
end

acc=n/length(te_label);
%end
% clear te_label
%acc(:,j)=accuracy;

% s1=sum(acc);
% ave1=s1/10;
% 
% s2=sum(a);
% ave2=s2/10;
% 


    

