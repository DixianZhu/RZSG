function demo(dataset, batch_num, batch_size, rep)
% Demo code:
%-------------------------shallow learning binary classification-----------
% input dataset
% batch_num: number of iteration for doing batch mode active learning.
% batch_size: batch size for active learning
% rep: number of times for repeating experiments.

% Author: Dixian Zhu, CS dept., Univ. of Iowa. 
%--------------------------------------------------------------------------
rng('shuffle');
fprintf('Loading Data\n');
dataPath=['./data/' dataset 'Train.mat'];
load(dataPath);
trX=X;
N=size(trX,1);
trX=[trX, ones(N,1)];
trY=y;
if ~exist('batch_num','var')
   batch_num=10; 
end
if ~exist('batch_size','var')
   batch_size=floor(N/10); 
end
if ~exist('rep','var')
   rep=10; 
end
load(['./data/' dataset 'Test.mat']);
teX=X;
Ntest=size(teX,1);
teX=[teX, ones(Ntest,1)];
teY=y;
trY=transLabel(trY);
teY=transLabel(teY);
%{
% uncomment this block if you want to compare with uniform sampling
%----------------tune regularization for C-SVM, libsvm---------------------
[model, bestC]=autoTune(trY,trX,10);
%--------------------------------------------------------------------------
uni=zeros(rep,batch_num);
for i=1:rep
    uni(i,:)=uniform(trX,trY,teX,teY,batch_num,batch_size,bestC);
    fprintf(['percent:' num2str(i/rep*100) '\n']);
end
uni=mean(uni,1)
%}

%-------------------tune regularization for RZSG---------------------------
[bestK,bestO]=autoTuneRadius(trY,trX);
fprintf(['best kappa is: ', num2str(bestK), ', best omega is: ' ,...
   num2str(bestO), '\n']);
%--------------------------------------------------------------------------
rzsg=zeros(1,batch_num);
for i=1:batch_num
   budget=i*batch_size;
   testAcc = 0;
   for j=1:rep
      testAcc=(testAcc*(j-1)+RZSG(trX,trY,teX,teY,budget,bestK,bestO))/j; 
   end
   rzsg(i)=testAcc
end
end
function y=transLabel(y)
class=unique(y);
if size(class,1)>2
    fprintf('This code is unable to deal with multiclass\n');
    exit;
end
 
index=union(find(class==1),find(class==-1));
if(size(index,2)<2)  %need to be transformed
    if class(index(1))==1
        trans=-1;
    else
        trans=1;
    end
    index=intersect(find(y~=1), find(y~=-1));
    if ~isempty(index) %double check need to be transformed
        fprintf(['transfer ' num2str(y(index(1))) ' to ' num2str(trans) '\n']);
        y(index)=trans;
        fprintf('transfer done\n');
    end
end
end