function B = ALM_optimize_C(X,Y,B,num,fdim,lambda,nbs,alpha)
% This routine solves the optimization problem of Eq.(5) in our SPL paper

tol = 1e-8;
maxIter = 1e6;
[d, n] = size(Y);
[m] = size(B,2);
[dx] = size(X,1);
rho = 1.1;
max_mu = 1e15;
mu = 1e-3;
Ys = Y(:,1:nbs);
Yt = Y(:,nbs+1:end);
Ys_sigma = cov(Ys',1);
Yt_sigma = cov(Yt',1);
Y_sigma_delta = Ys_sigma - Yt_sigma;
Ys_mean = mean(Ys,2);
Yt_mean = mean(Yt,2);
Y_mean_delta = Ys_mean - Yt_mean;
%% Initializing optimization variables
% intialize
% alpha is mu in the paper
% mu is kappa in the paper
K = B;% K is Q in the paper
% B = sparse(m,d);
T1 = zeros(d,m);
T2 = zeros(d,m);

%% Start main loop
iter = 0;

while iter<maxIter
    iter = iter + 1;
    %update P  
    if d > n
        Y_tmp = Y;
        Y1 = [Y_tmp,sqrt(alpha)*Y_mean_delta,sqrt(alpha)*Y_sigma_delta*K];
        [Q,R] = qr(Y1,0);
        [Q1,D,Q1] = svd(R*R');
        U = Q*Q1;
    else
        [U,D,U] = svd(Y*Y'+ alpha*(Y_mean_delta*Y_mean_delta'+Y_sigma_delta*(K*K')*Y_sigma_delta));
    end

    if d > n
        tmp = 2*diag(D)/mu + ones(size(D,1),1);
        inv_tmp = tmp.^-1;
        P = U*(diag(inv_tmp)*(U'*(B+(2*Y*X'-T1)/mu))) + ((B+(2*Y*X'-T1)/mu)-U*(U'*(B+(2*Y*X'-T1)/mu)));
    else
        tmp = 2*diag(D)/mu + ones(d,1);
        inv_tmp = tmp.^-1;
    %     inv_y = U*diag(inv_tmp)*U';
        P = U*(diag(inv_tmp)*(U'*(B+(2*Y*X'-T1)/mu)));
    end
    %update K
    if d > n
        Y_tmp = Y_sigma_delta*P;
        [QQ,RR] = qr(Y_tmp,0);
        [QQ1,DD,QQ1] = svd(RR*RR');
        UU = QQ*QQ1;
    else
        [UU,DD,UU] = svd(Y_sigma_delta*(P*P')*Y_sigma_delta);
    end
    
    if d > n
        tmp = 2*diag(DD)/mu + ones(size(DD,1),1);
        inv_tmp = tmp.^-1;
        K = UU*(diag(inv_tmp)*(UU'*(B-T2/mu))) + ((B-T2/mu)-UU*(UU'*(B-T2/mu)));
    else
        tmp = 2*diag(DD)/mu + ones(d,1);
        inv_tmp = tmp.^-1;
    %     inv_y = U*diag(inv_tmp)*U';
        K = UU*(diag(inv_tmp)*(UU'*(B-T2/mu)));
    end

    %update B

    W = (P+K)/2+(T1+T2)/mu;
    B = solve_l1l2(W,num,fdim,lambda/(2*mu));
    

    leq1 = P-B;
    leq2 = K-B;
    stopC1 =max(max(abs(leq1)));
    stopC2 =max(max(abs(leq2)));
    if iter==1 || mod(iter,50)==0 || (stopC1<tol && stopC2<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC1,'%2.3e'),' and ',num2str(stopC2,'%2.3e')]);
    end
    if stopC1<tol && stopC2<tol
        disp('the optimization of C is done.');
        break;
    else
        T1 = T1 + mu*leq1;
        T2 = T2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end

