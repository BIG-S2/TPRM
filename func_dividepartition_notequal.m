function [Partition, index_partition]=func_dividepartition_notequal(Y_new,SP1,SP2,SP3)

% Inputs: Y_new : 4D tnsor to be partitioned
%        SP : the size of the partitions

% Outputs: Partition: a cell with each cell element a partition of size SP1 x
%                     SP2 x SP3 x N
%          index_partition: index with partitions that are not zero for all
%                           elements

% Example: if Y_new is p1 x p2 x p3 and SP1 = sp1, SP2 = sp2, SP3 = sp3,
% each side will be divide by pj/spj  producing partitions of size SP1 x
% SP2 x SP3


[dimP1,dimP2,dimP3,~]=size(Y_new);
aux=(dimP1/SP1)*(dimP2/SP2)*(dimP3/SP3);
SP01=dimP1/SP1;
SP02=dimP2/SP2;
SP03=dimP3/SP3;
Partition=cell(aux,1);
count=1;
for i=1:SP01
    for j=1:SP02
        for k=1:SP03
          Partition{count}= Y_new(SP1*(i-1)+1:i*SP1,SP2*(j-1)+1:j*SP2,SP3*(k-1)+1:k*SP3,:);
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
        