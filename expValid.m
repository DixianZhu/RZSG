function Q=expValid(Q)
if max(Q)>300
	Q=Q-(max(Q)-300);
end
end
