function [Partition, index_partition]=func_dividepartition(Y_new,SP)

%Inputs: Y_new : 4D tensor to be partitioned
%        SP : the size of the partitions


% Example if Y_new is 96 x 96 x 96 and SP = 4, each side will be divide by
% 24, producing partitions of size 6 x 6 x 6

% Note: this code is for partitioning the images with same size, if that is
% not the case, use func_dividepartition_notequal instead
dimP=size(Y_new,1);
aux=(dimP/SP)^3;
SP0=dimP/SP;
Partition=cell(aux,1);
count=1;
for i=1:SP0
    for j=1:SP0
        for k=1:SP0
          Partition{count}= Y_new(SP*(i-1)+1:i*SP,SP*(j-1)+1:j*SP,SP*(k-1)+1:k*SP,:);
          count=count+1;
        end
    end
end

%Many partitions are 0
count=0;
auxtest=zeros(aux,1);
for tes=1:aux
    if sum(sum(sum(sum(Partition{tes}))))==0
        count=count+0;
        auxtest(tes,1)=tes;
    else
        count=count+1;
        auxtest(tes,1)=0;
    end
end

 index_partition=find(auxtest==0);       
        