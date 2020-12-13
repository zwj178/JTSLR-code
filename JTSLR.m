    function [P,Q] =TSRL(Y_src,X_src,X_tar_train,Y_tar_train,options)

	lambda = options.lambda;              %% lambda for the regularization
	dim = options.dim;                    %% dim is the dimension after adaptation, dim <= m
	gamma= options.gamma;
    alpha= options.alpha;
    T=options.T;
     
    %% 
    Y_tar_pseudo=[];
    %%
	%%% MMD
    X = [X_src,X_tar_train];
   % X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns = size(X_src,2);
	nt = size(X_tar_train,2);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	 C = length(unique(Y_src));
     M = e * e'*C ;
     M= M / norm(M,'fro');
     H = eye(n) - 1/n * ones(n,n);  
     [Q,~] = eigs(alpha*X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
     Zs=Q'*X_src;
     Zt=Q'*X_tar_train;
     knn_model = fitcknn(Zs',Y_src,'NumNeighbors',1);
     Y_tar_pseudo = knn_model.predict(Zt');     
      Z=[Zs,Zt];
%           model= libsvmtrain(Y_src,Zs','-s 0 -t 0 -c 1 -g 1  ');
%           [predict_label] = libsvmpredict(Y_tar_train,Zt', model);  
%           Y_tar_pseudo=predict_label;
     
    %label matrix
        Y=[Y_src;Y_tar_pseudo];
         nbclass = unique(Y);
         rankk = length(nbclass);
         Y_matrix = zeros(rankk,length(Y));
     for i = 1:length(nbclass)
         labels = (Y== nbclass(i));
         labels = double(labels);
         Y_matrix(i,:) = labels;
     end
     options.ReducedDim =5;
     [P1,~] = PCA1(Z',options);
     P1=P1';
%      P1=ones(rankk,dim);
%      B = Y_matrix;
%      B(Y_matrix==0) = -1;
%      Ys_matrix=Y_matrix(:,1:ns);
%    Ytr_matrix=zeros(rankk,nt);
     Ys_matrix=Y_matrix(:,1:ns);
     Ytr_matrix=Y_matrix(:,ns+1:end);
     for iter=1:T    
     %update P
     Y_matrix=[Ys_matrix,Ytr_matrix];   
     if iter==1
         P=P1;
     else
      [U1,S1,V1] = svd(Y_matrix*X'*Q,'econ'); 
       P = U1*V1';
     end
%       end      
       M = e * e'*C ;
       N = 0;
       for c = reshape(unique(Y_src),1,C)
			e = zeros(n,1);
			e(Y_src==c) = 1 / length(find(Y_src==c));
			e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
			e(isinf(e)) = 0;
			N = N + e*e';
        end
         M = M + N;
         M = M / norm(M,'fro');
       
      %update Q
       Q1 = alpha*X*M*X'+X*X'+lambda*eye(m);
       Q2 = X*Y_matrix'*P;  
       Q =  Q1\Q2;       
     
           %  construct graph 
        options.t=1;
        options.NeighborMode = 'Supervised';
        options.label=[Y_src;Y_tar_pseudo];
        options.k = 3;
        [W12,W21]= CW(X_src',X_tar_train',options);
        options.t=1;
         options.NeighborMode='Supervised';
         options.gnd = Y_src;
         options.WeightMode='HeatKernel';
         options.k=5;
        W11 = constructW(X_src',options);
        options.gnd = Y_tar_pseudo;
        W22 = constructW(X_tar_train',options);
        W1=[W11,W12];        
        W2=[W21,W22];
        W=[W1;W2];
        W=full(W);
        D = full(sum(W,2));
        D=diag(D);
        L=D-W; 
%        L=L';
     %update Ytr            
%        if gamma~=0
         Y1=gamma*L+eye(n);
         Y2=P*Q'*X;
         Y_matrix=Y2/Y1;
          for ii=1:n
              Y_matrix(:,ii)=Y_matrix(:,ii)/sum(Y_matrix(:,ii));
          end
          
       Ytr_matrix=Y_matrix(:,ns+1:end);      
       [Max_value,Y_tr_label] = max(Ytr_matrix);
       Y_tar_pseudo=Y_tr_label';
           z=gamma*trace(Y_matrix*L*Y_matrix');  
          %%
       leq=Y_matrix-P*Q'*X;
       obj(iter) = norm(leq,'fro')+alpha*trace(Q'*X*M*X'*Q)+gamma*trace(Y_matrix*L*Y_matrix')+lambda*norm(Q, 'fro' );
     end
     
end