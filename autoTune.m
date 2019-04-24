function [W, minC]=autoTune(y,X,s)
% Tune SVM with 5-folds cross-validation
C=[0.001,0.01,0.1,1,10,100,1000];
[N,D]=size(X);
M=floor(N/5); % validation size
minE=realmax;
minC=C(1);
for i=1:size(C,2)
	trEr=0;
	vaEr=0;
	perm=randperm(N);
	for j=1:5
		vaID=perm(((j-1)*M+1):j*M);
		trID=setdiff(1:N,vaID);
		trX=X(trID,:);
		trY=y(trID);
		vaX=X(vaID,:);
		vaY=y(vaID);
		if s<10
			model=train(trY,trX,['-q -s ' num2str(s) ' -c ' num2str(C(i))]);
		else
			model=svmtrain(trY,trX,['-q -t ' num2str(s-10) ' -c ' num2str(C(i))]);
		end
		if s<10
			[~,trAcc,~]=predict(trY,trX,model,'-q');
			[~,vaAcc,~]=predict(vaY,vaX,model,'-q');
		else	
			[~,trAcc,~]=svmpredict(trY,trX,model,'-q');
			[~,vaAcc,~]=svmpredict(vaY,vaX,model,'-q');
		end
		trEr=trEr+100-trAcc(1);
		vaEr=vaEr+100-vaAcc(1);
	end
	trEr=trEr/5;
	vaEr=vaEr/5;
	if minE>vaEr
		minE=vaEr;
		minC=C(i);
	end
	fprintf(['C=' num2str(C(i)) ', training error: ' num2str(trEr) '%%, validation error: ' num2str(vaEr) '%% \n']);
end
fprintf(['The best C is:' num2str(minC) '\n']);
if s<10
	W=train(y,X,['-q -s ' num2str(s) ' -c ' num2str(minC)]);
else
	W=svmtrain(y,X,['-q -t ' num2str(s-10) ' -c ' num2str(minC)]);
end
end
