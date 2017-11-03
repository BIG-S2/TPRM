%This code runs Scenario 7 and 8 Section 3.1 described in the paper:
% TPRM:  Tensor partition regression models with applications in  
% imaging biomarker detection (Michelle F. Miranda, Hongtu Zhu and Joseph
% G. Ibrahim

% It requires the matlab toolboxes: 1) k-wave-toolbox-version-1.1.1 
% and toolbox 2) tools-for-nifti-and-analyze-image; and 3) tensor_toolbox


%--------------------------------
%   DATA GENERATION
%--------------------------------

SignalConst=65; %Scenario 4
%SignalConst=50; %Scenario3
R=20;

rand('seed',10);
randn('seed',10);

% load template $\mathcal{G}_0$
% Need toolbox tools-for-nifti-and-analyze-image
template=load_nii('Template_DS2');
A=template.img;
[iv,jv,kv]=ind2sub(size(A), find(A>0));
NoP=size(iv,1);
dimx=64;
dimy=64;
dimz=50;
ssize=[dimx, dimy,dimz];
MaskVoxels=sub2ind(ssize,iv,jv,kv);

%Generate signal $\mathcal{X}_0$
%Generate signal
B1=zeros(64,4);
B2=zeros(64,4);
B3=zeros(50,4);
for j=1:6
   
    B1(18+j,1)=sin(j*pi/14);
    B2(18+j,1)=sin(j*pi/14);
    B3(23+j,1)=sin(j*pi/14);
    
    B1(23+j,2)=sin(j*pi/14);
    B2(23+j,2)=sin(j*pi/14);
    B3(23+j,2)=sin(j*pi/14);
end
for j=1:4
    B1(32+j,3)=sin(j*pi/14);
    B2(32+j,3)=sin(j*pi/14);
    B3(32+j,3)=sin(j*pi/14);
    
    B1(38+j,4)=1.3*sin(j*pi/14);
    B2(38+j,4)=1.3*sin(j*pi/14);
    B3(23+j,4)=1.3*sin(j*pi/14);
end
TB=ktensor([1 1 1 1]',B1,B2,B3);
%TB=ktensor(1,B1,B2,B3);

ball=double(tensor(TB));
auxball=ball;
auxball(MaskVoxels)=ball(MaskVoxels);
%For visualization of the signal
%SignalSce=make_nii(auxball);
%view_nii(SignalSce)

%Generate short-term noise

N=200;
cc=70;
% N images with short-term noise
noise_short1=zeros(dimx*dimy*dimz,N);
noise=cc*randn(dimx*dimy*dimz,N);
 % short-term noise
 for k=1:dimz    
    for i=1:dimx       
       for j=1:dimy  
            if k<dimz && i<dimx && j<dimy
               noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=(noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:)+...
               noise(i+1+(j-1)*dimx+(k-1)*dimx*dimy,:)+noise(i+j*dimx+(k-1)*dimx*dimy,:)+noise(i+(j-1)*dimx+k*dimx*dimy,:))/4;
            elseif k==dimz && i<dimx && j<dimy
               noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=(noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:)+...
               noise(i+1+(j-1)*dimx+(k-1)*dimx*dimy,:)+noise(i+j*dimx+(k-1)*dimx*dimy,:))/3;
            elseif k<dimz && i==dimx && j<dimy
               noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=(noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:)+...
               noise(i+j*dimx+(k-1)*dimx*dimy,:)+noise(i+(j-1)*dimx+k*dimx*dimy,:))/3;
            elseif k<dimz && i<dimx && j==dimy
               noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=(noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:)+...
               noise(i+1+(j-1)*dimx+(k-1)*dimx*dimy,:)+noise(i+(j-1)*dimx+k*dimx*dimy,:))/3;
           elseif k<dimz && i==dimx && j==dimy
             noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=(noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:)+...
                                              noise(i+(j-1)*dimx+k*dimx*dimy,:))/2;
           elseif k==dimz && i<dimx && j==dimy
             noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=(noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:)+...
                               noise(i+1+(j-1)*dimx+(k-1)*dimx*dimy,:))/2;
           elseif k==dimz && i==dimx && j<dimy  
             noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=(noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:)+...
                                              noise(i+j*dimx+(k-1)*dimx*dimy,:))/2;
            else
             noise_short1(i+(j-1)*dimx+(k-1)*dimx*dimy,:)=noise(i+(j-1)*dimx+(k-1)*dimx*dimy,:);
            end           
      end        
    end    
 end

 
noise_short_term=reshape(noise_short1,dimx,dimy,dimz,N);

% Generate 3D data $X_i=\mathcal{G}_0 + y_i \mathcal{X}_0 + \epsilon_i$
%Need toolbox tensor_toolbox

TImg2=zeros(dimx,dimy,dimz,N);%Store tensor data
Y=binornd(1,0.5,N,1);
TI=zeros(dimx,dimy,dimz,N); 


for i=1:N
    if Y(i,1)==0
        TI(:,:,:,i)=double(A)+SignalConst*auxball+noise_short_term(:,:,:,i);
    else
        TI(:,:,:,i)=double(A)+noise_short_term(:,:,:,i);
    end
end
for i=1:N
    TImg2(:,:,:,i)=TI(:,:,:,i)-mean(TI,4); %Centralize the tensor data
end
clear TI

% Data concatanation
NoVoxels=64*64*50;
MatrixT=zeros(N,NoVoxels);
for i=1:N
    aux=TImg2(:,:,:,i);
    MatrixT(i,:)=reshape(aux,1,NoVoxels);
end


%Diving the data into partitions
dimP=64;
dim2=50;

SP0=4; %Divides dim 1 and 2 into 4 parts
SP1=2; %Divides dim 3 into 2 parts
%Partitions are now size dimP/SP0 x dimP/SP0 x dim2/SP1 (16x16x25)

auxpp=(dimP/(dimP/SP0))^2*(dim2/(dim2/SP1));%Total number of partitions
Partition=cell(auxpp,1);
Y_new=TImg2;
count=1;
for i=1:SP0
    for j=1:SP0
        for k=1:SP1
          Partition{count}= Y_new((dimP/SP0)*(i-1)+1:i*(dimP/SP0),(dimP/SP0)*(j-1)+1:j*(dimP/SP0),(dim2/SP1)*(k-1)+1:k*(dim2/SP1),:);
          count=count+1;
        end
    end
end

% Exclude partitions formed by 0 only (in this example none)
count=0;
auxtest=zeros(auxpp,1);
for tes=1:auxpp
    if sum(sum(sum(sum(Partition{tes}))))==0
        count=count+0;
        auxtest(tes,1)=tes;
    else
        count=count+1;
        auxtest(tes,1)=0;
    end
end
index_partition=find(auxtest==0);      
npart=size(index_partition,1);


%%% SPLITTING THE DATA 

NFolds=10;
%CVP=cvpartition(N,'kFold',NFolds); %% 10-fold cv
%save('Sim_10fold_sets','CVP');
load('Sim_10fold_sets')
ntest=20;
ntrain=180;



PAccuracy_fpca=zeros(NFolds,1);
PAccuracy_NoPartition=zeros(NFolds,1);
PAccuracy_PartitionGLM=zeros(NFolds,1);
PAccuracy_PartitionTPRM=zeros(NFolds,1);


False_Positivesfpca=zeros(NFolds,1);
False_Negativesfpca=zeros(NFolds,1);
False_PositivesNoPartition=zeros(NFolds,1);
False_NegativesNoPartition=zeros(NFolds,1);
False_PositivesTPRM=zeros(NFolds,1);
False_NegativesTPRM=zeros(NFolds,1);
False_PositivesGLM=zeros(NFolds,1);
False_NegativesGLM=zeros(NFolds,1);

for sim=1:NFolds
%tic
indexTra=training(CVP,sim);
indexTest=test(CVP,sim);

Y_train=Y(indexTra,1);
Y_test=Y(indexTest,1);

MatrixT_test=MatrixT(indexTest,:);
MatrixT_train=MatrixT(indexTra,:);

TImg2_train=TImg2(:,:,:,indexTra);
TImg2_test=TImg2(:,:,:,indexTest);

True_positives=find(Y_test==1);
True_negatives=find(Y_test==0);


% 1) FPCA

[SCORE,latent,COEFF] = svd(MatrixT_train,'econ');
%[COEFF2,latent2,SCORE2]=princomp(MatrixT_train,'econ'); %COEFF and COEFF2 are equivalent

%We fixed R but it can be computed  to retain 99.9% of the cariance or any
%percentage. Just be careful on convergence and to avoid overfitting

%scree_train=[0;cumsum(diag(latent).^2)/sum(diag(latent).^2)];
%idx=0:size(latent,1);
%R=min(idx(scree_train>=0.8)); 
 
Cov_pca_train=MatrixT_train*COEFF(:,1:R);
Cov_pca_test=[ones(ntest,1) MatrixT_test*COEFF(:,1:R)];
b_pca =glmfit(Cov_pca_train,Y_train,'binomial','link','probit');
aux_pca=(normcdf(Cov_pca_test*b_pca))>=0.5;
PAccuracy_fpca(sim,1)=sum(Y_test==aux_pca)/ntest;
False_Positivesfpca(sim,1)=1-sum(aux_pca(True_positives))/size(True_positives,1);
False_Negativesfpca(sim,1)=1-sum(1-aux_pca(True_negatives))/size(True_negatives,1);

% 2) Tensor Decomposition 

TCP=cp_als(tensor(TImg2_train),R,'printitn',0);
V=func_squares(TCP,4);
XR1_train= mttkrp(tensor(TImg2_train),TCP.U,4)*pinv(V)*diag(1./TCP.lambda);
% XR1_train is equivalent to TCP.U{4}
XR1Test= mttkrp(tensor(TImg2_test),TCP.U,4)*pinv(V)*diag(1./TCP.lambda);
XR1_test=[ones(ntest,1) XR1Test];
[bt DEVt statst] = glmfit(XR1_train,Y_train,'binomial','link','probit');
aux_tensor=(normcdf(XR1_test*bt))>=0.5;
PAccuracy_NoPartition(sim,1)=sum(Y_test==aux_tensor)/ntest; 
False_PositivesNoPartition(sim,1)=1-sum(aux_tensor(True_positives))/size(True_positives,1);
False_NegativesNoPartition(sim,1)=1-sum(1-aux_tensor(True_negatives))/size(True_negatives,1);


% 3) TPRM
RR=20; %Rank inside the partitions, it can be RR=R;
indexTra=training(CVP,sim);
indexTest=test(CVP,sim);
indexTest_aux=find(indexTest==1);

Y_train=Y(indexTra,1);
Y_test=Y(indexTest,1);

%Partition model

XMat=cell(npart,1);
XMat2=cell(npart,1);
AMat=cell(npart,1);
BMat=cell(npart,1);
CMat=cell(npart,1);
Lamb=cell(npart,1);
for i=1:npart
    aux=index_partition(i,1);
    YT_train=tensor(Partition{aux}(:,:,:,indexTra));
    YT_test=tensor(Partition{aux}(:,:,:,indexTest));
    T=cp_als(YT_train,RR,'printitn',0);
    V=pinv(func_squares(T,4));
    XMat{i}=  mttkrp(YT_train,T.U,4)*V*diag(1./T.lambda);
    XMat2{i}= mttkrp(YT_test,T.U,4)*V*diag(1./T.lambda);
   
end


%Creating the matrix L
NR=size(XMat{1},2);
k0=NR*npart;

X1s=[];
X2s=[];
    for i=1:npart
        X1s=[X1s XMat{i}];
        X2s=[X2s XMat2{i}];
    end

%Removing NaN values if any
aux11=zeros(k0,1);
for ii=1:k0
aux11(ii,1)=isnan(sum(X1s(:,ii),1));
end
indexnan=find(aux11==0);
X1=X1s(:,indexnan);
X2=X2s(:,indexnan);

[N P] = size(X1);
MeanCol=mean(X1,1);
MeanCol2=mean(X2,1);
L=zeros(size(X1));
L2=zeros(size(X2));
for j=1:P
    L(:,j)=X1(:,j)-MeanCol(1,j);
    L2(:,j)=X2(:,j)-MeanCol2(1,j);
end

[SCORE2,latent2,COEFF2] =svd(L,'econ');

Cov_train=L*COEFF2(:,1:R);
Cov_test=L2*COEFF2(:,1:R);
Cov_test2=[ones(ntest,1) Cov_test];
[b2,~,~] = glmfit(Cov_train,Y_train,'binomial','link','probit');
aux_part=(normcdf(Cov_test2*b2))>=0.5;
PAccuracy_PartitionGLM(sim,1)=sum(Y_test==aux_part)/ntest; 
False_PositivesGLM(sim,1)=1-sum(aux_part(True_positives))/size(True_positives,1);
False_NegativesGLM(sim,1)=1-sum(1-aux_part(True_negatives))/size(True_negatives,1);


%Here it starts the MCMC

NSim=10000; %Size of the chain
alpha_pi=0.5;
beta_pi=0.5;
pi_delta=betarnd(alpha_pi,beta_pi);
deltaj=binornd(1,pi_delta,R,1);
betamat=zeros(R,NSim);
f0=find(Y_train==0);
f1=find(Y_train==1);
ydraw=zeros(size(Y_train));

G=Cov_train;
ydraw(f0,1)=trunc_norm(G(f0,:)*b2(2:end),0);
ydraw(f1,1)=trunc_norm(G(f1,:)*b2(2:end),1);
betadraw=b2(2:end); % or betadraw=normrnd(0,1,K1,1) for example;
tic
for j=1:NSim   
   %Sample b
   [deltaj,betadraw,pi_delta]=sample_deltaj(G,ydraw,pi_delta,betadraw,R);
   betamat(:,j)=betadraw;
   %Generate latent response
   ydraw(f0)=trunc_norm(G(f0,:)*betadraw,0);
   ydraw(f1)=trunc_norm(G(f1,:)*betadraw,1);
   
end
PredictedY=(normcdf(Cov_test*mean(betamat(:,5001:end),2)))>=0.5;
PAccuracy_PartitionTPRM(sim,1)=sum(Y_test==PredictedY)/ntest; 

False_PositivesTPRM(sim,1)=1-sum(PredictedY(True_positives))/size(True_positives,1);
False_NegativesTPRM(sim,1)=1-sum(1-PredictedY(True_negatives))/size(True_negatives,1);

%toc


end %10-fold

TableAccuracy=[PAccuracy_fpca PAccuracy_NoPartition PAccuracy_PartitionTPRM];
TableFalsePositive=[False_Positivesfpca False_PositivesNoPartition False_PositivesTPRM];
TableFalseNegative=[False_Negativesfpca False_NegativesNoPartition False_NegativesTPRM];

MeanAccuracy=[mean(PAccuracy_fpca) mean(PAccuracy_NoPartition) mean(PAccuracy_PartitionTPRM)];
MeanFalsePositive=[mean(False_Positivesfpca) mean(False_PositivesNoPartition) mean(False_PositivesTPRM)];
MeanFalseNegative=[mean(False_Negativesfpca) mean(False_NegativesNoPartition) mean(False_NegativesTPRM)];


save(strcat('Results_Scenario8_R',num2str(R),'_RR',num2str(RR),'.mat'))

