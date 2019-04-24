function [bestK, bestO]=autoTuneRadius(y,X,gamma,eta)
%kappa=[0.001,0.01,0.1,1,10,100,1000];
kappa=[0.01,0.1,1];
omega=kappa;
[N,D]=size(X);
if N*D > 1e6 || N>1e5
	subN=1000;
	fprintf(['N: ' num2str(N) ', D: ' num2str(D) ', sub-sample subN:' num2str(subN) '\n']);
	perm=randperm(N);
	X=X(perm(1:subN),:);
	y=y(perm(1:subN));
end
N=size(X,1);
M=floor(N/5); % validation size
n=N-M;
bestAcc=0;
bestK=kappa(1);
bestO=omega(1);
T=100*n;
if ~exist('eta','var')||eta==0
    eta=2*sqrt(2)/(sqrt(max(sum(X.^2)))*sqrt(T));
end
if ~exist('gamma','var')||gamma==0
    gamma=sqrt(log(N))/((1+max(max(X)))*sqrt(T));
end
for i=1:size(kappa,2)
    for k=1:size(omega,2)
        acc=0;
        perm=randperm(N);
        for j=1:5
            vaID=perm(((j-1)*M+1):j*M);
            trID=setdiff(1:N,vaID);
            trX=X(trID,:);
            trY=y(trID);
            vaX=X(vaID,:);
            vaY=y(vaID);
            pid=trY==1;
            nid=trY==-1;
            % training --------------------------------------------------
            w=zeros(1,D);
            W=w;
            pscore=ones(1,n);
            pscore=pscore/norm(pscore,1);
            for t=1:T
                sigG=sign(max(0,1-trY'.*(w*trX')));
                sigG=sigG.*pscore;
                sigG=sigG.*trY';
                stepSize=eta;
                % update w-----------------------
                w = w + stepSize*sigG*trX;
                if norm(w)>kappa(i)
                    w = (w/norm(w))*kappa(i);
                end
                % -------------------------------
                W=(W*(t-1)+w)/t;
                % update pscore------------------
                Q=computeQ(trX,w,pid,nid);
                stepSize=gamma;         
                
                % regularizer form for prob.
                pscore=pValid(pscore.*exp(expValid(stepSize*(Q-omega(k)*(1+log(pscore))))));
                
                % proximal mapping form for prob.
%                pscore=pValid(pscore.*exp(expValid(stepSize*Q)));
%                pscore=pscore.^(1/(1+stepSize*omega(k)));

                pscore=pscore/norm(pscore,1);          
                % -------------------------------
            end
            %------------------------------------------------------------
            
            % testing ---------------------------------------------------
            acc=acc+nnz(max(vaY'.*(W*vaX'),0))./M;
            %------------------------------------------------------------
        end
        if bestAcc<acc
            bestAcc=acc
            bestK=kappa(i);
            bestO=omega(k);
        end
    end
end

end

function Q=computeQ(trX,w,pid,nid)
    Q=w*trX';
    nQ=exp(Q);
    lnQ=1+nQ;
    loss_nQ=max(0,1+Q);
    loss_pQ=max(0,1-Q);
    nQ=1./lnQ;
    nQ(pid)=0;
    nQ(nid)=1;
    Q=nQ.*loss_nQ+(1-nQ).*loss_pQ;
end
