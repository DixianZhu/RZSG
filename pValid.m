function p=pValid(p)	
	id=find(p<realmin);
	p(id)=realmin;
end
