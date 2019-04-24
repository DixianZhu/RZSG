function testAcc = RZSG(trX, trY, teX, teY, budget, kappa, omega, gamma, eta)
% Robust Zero-Sum Gaming active learning method
% Author: Dixian Zhu
% Zhu, D., Li, Z., Wang, X., Gong, B. & Yang, T.. (2019).
% A Robust Zero-Sum Game Framework for Pool-based Active Learning. 
% Proceedings of Machine Learning Research, in PMLR 89:517-526


T=1000*budget;
[N,D]=size(trX);
if ~exist('gamma','var')||gamma==0
    gamma=sqrt(log(N))/((1+kappa*max(max(trX)))*sqrt(T));
end
if ~exist('eta','var')||eta==0 
    eta=2*sqrt(2)*kappa/(sqrt(max(sum(trX.^2)))*sqrt(T));
end
L=[];
w=zeros(1,D);
W=w;
pscore=ones(1,N);
pscore=pscore./norm(pscore,1);
sizeL=0;
for t=1:T
    if sizeL < budget
        H=randsample(1:N,1,true,pscore);
        L=union(L,H);
        sizeL=size(L,2);
    end
    lscore = pscore(L);
    lscore = lscore./norm(lscore,1);
    X=trX(L,:);
    y=trY(L);
    sigG=sign(max(0,1-y'.*(w*X')));
    sigG=sigG.*lscore; % expectation term for gradient
    sigG=sigG.*y';
    w = w + eta*sigG*X;
    if norm(w)>kappa
        w = (w/norm(w))*kappa;
    end
    W=(W*(t-1)+w)/t;
    Q=computeQ(w,trX,trY,L);
    % regularizer update form
    pscore=pValid(pscore.*exp(expValid(gamma*(Q-omega*(1+log(pscore))))));
    
    % proximal mapping update form
    %pscore=pValid(pscore.*exp(expValid(stepSize*Q)));
    %pscore=pscore.^(1/(1+stepSize*omega));
    
    pscore=pscore/norm(pscore,1);
end
testAcc=nnz(max(teY.*(teX*W'),0))/size(teY,1)
size(L)
end
function Q=computeQ(w,trX,trY,labeled)
    Q=w*trX';
    nQ=exp(Q);
    lnQ=1+nQ;
    lgnQ=max(0,1+Q);
    lgpQ=max(0,1-Q);
    nQ=1./lnQ;
    pid=labeled(trY(labeled)==1);
    nQ(pid)=0;
    nid=labeled(trY(labeled)==-1);
    nQ(nid)=1;
    Q=nQ.*lgnQ+(1-nQ).*lgpQ;
end
