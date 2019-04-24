function accuracy=uniform(trX, trY, teX, teY, batch_num, batch_size, bestC)
[N,D]=size(trX);
U=1:N;
L=[];
accuracy=zeros(1,batch_num);
for i=1:batch_num
    batch=randsample(U,batch_size,false);
    U=setdiff(U, batch);
    L=[L, batch];
    X=trX(L,:);
    y=trY(L,:);
    model=svmtrain(y,X,['-q -t 0 -c ' num2str(bestC)]); % use libsvm 
    [~,acc,~]=svmpredict(teY,teX,model,'-q');
    accuracy(i)=acc(1)/100;
end
end