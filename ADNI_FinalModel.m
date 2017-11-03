% Code for running the model for the ADNI dataset
% TPRM:  Tensor partition regression models with applications in  
% imaging biomarker detection (Michelle F. Miranda, Hongtu Zhu and Joseph
% G. Ibrahim

% It requires the matlab toolboxes: 1) tools-for-nifti-and-analyze-image; 
%   and 2) tensor_toolbox
% It calls the functions: func_dividepartition, sample_deltaj, trunc_norm, functionIC


%Load the data
load('/Users/miranda/Documents/MATLAB/ADNI_RavensMap/Tensor_ADNI_4_red.mat')
load('/Users/miranda/Documents/MATLAB/ADNI_RavensMap/Tensor_ADNI_3_red.mat')
load('/Users/miranda/Documents/MATLAB/ADNI_RavensMap/Tensor_ADNI_2_red.mat')
load('/Users/miranda/Documents/MATLAB/ADNI_RavensMap/Tensor_ADNI_1_red.mat')

%%n1=100;
%n2=81;
%n3=100;
%n4=121;
%Data was split into 4 files due to memory concerns, the first 2 files are
%     AD patients, the last 2 files are normal controls
YTotal=zeros(96,96,96,402);
YTotal(:,:,:,1:100)=Tensor_ADNI_1;
YTotal(:,:,:,101:181)=Tensor_ADNI_2;
YTotal(:,:,:,182:281)=Tensor_ADNI_3;
YTotal(:,:,:,282:n1+n2+n3+n4)=Tensor_ADNI_4;


%Partition the tensor
[Partition, index_partition]=func_dividepartition(YTotal,6);
clearvars -except Partition index_partition

R=5;
npart=size(index_partition,1);

% Tensor Decomposition
X1Mat=cell(npart,1);
A1Mat=cell(npart,1);
B1Mat=cell(npart,1);
C1Mat=cell(npart,1);
L1amb=cell(npart,1);

for i=1:npart
%    tic
    aux=index_partition(i,1);
    YT=tensor(Partition{aux});
    T1=cp_als(YT,R);
    X1Mat{i}=T1{4};
    A1Mat{i}=T1{1};
    B1Mat{i}=T1{2};
    C1Mat{i}=T1{3};
    L1amb{i}=T1.lambda;
%   toc
end



% Second stage modeling
XR1=[]; %Matrix L in the paper
index_conv_problems=[];
for j=1:npart
 if isnan(L1amb{j})
    XR1=[XR1];
    index_conv_problems=[index_conv_problems,j];
 else
    XR1=[XR1 X1Mat{j}];
 end
end

% Principal components on matrix L
[COEFF0 G0 latent0]=princomp(XR1,'econ');
%scree_train=[0;cumsum(latent0)./sum(latent0)];
%idx=0:size(latent0,1);
%K=min(idx(scree_train>=0.99)); %Take K that explain 99% of variability or 95%
K=50; %Obs: Maximum K such that we still have conv.
X0=XR1*COEFF0(:,1:K);
[N P]=size(X0);



%%%%%%%%%%%%%%%%%%%%
%     MODELING
%%%%%%%%%%%%%%%%%%%%


Y=[ones(181,1); zeros(221,1)]; %Response variable
NSim=150000;
alpha_pi=0.5;
beta_pi=0.5;
pi_delta=betarnd(alpha_pi,beta_pi);
deltaj=binornd(1,pi_delta,K,1);

betamat=zeros(K,NSim);

f0=find(Y==0);
f1=find(Y==1);
betadraw=normrnd(0,0.1,K+1,1);
%[betadraw, dev,stats]=glmfit(X0C,Y,'binomial', 'link', 'probit');


X0C=zeros(size(X0));%Standardized matrix to be used on the MCMC
MX0=mean(X0,1);
SX0=std(X0,1,1);
for i=1:N
X0C(i,:)=(X0(i,:)-MX0)./SX0;
end
G=X0C;

%Initial latent variables
ydraw(f0,1)=trunc_norm(G(f0,:)*betadraw(2:end),0);
ydraw(f1,1)=trunc_norm(G(f1,:)*betadraw(2:end),1);
betadraw=betadraw(2:end);

%MCMC starts here
tic
for j=1:NSim
%Sample b
   [deltaj,betadraw,pi_delta]=sample_deltaj(G,ydraw,pi_delta,betadraw,K);
   betamat(:,j)=betadraw;
%Generate latent response
   ydraw(f0)=trunc_norm(G(f0,:)*betadraw,0);
   ydraw(f1)=trunc_norm(G(f1,:)*betadraw,1);
   
end
toc


%Getting the final partition index

%load('Partition6_R5.mat')
index_partition2=index_partition;
aux=index_conv_problems';
index_partition2(aux,1)=0;
index_partition2(index_partition2==0)=[];
npart1=size(index_partition2,1);

indexnew=1:1:npart1;
indexnew(index_conv_problems)=[];
Lamb=cell(npart1,1);
AMat=cell(npart1,1);
BMat=cell(npart1,1);
CMat=cell(npart1,1);
 for i=1:npart1
     auxi=indexnew(1,i);
     Lamb{i}=L1amb{auxi};
     AMat{i}=A1Mat{auxi};
     BMat{i}=B1Mat{auxi};
     CMat{i}=C1Mat{auxi};
 end
%Obtaining Projections
%Projecting important basis
ImportantBeta=zeros(size(betamat,1),1);
for ii=1:size(betamat,1)
    ImportantBeta(ii,1)=functionIC(betamat(ii,5000:50:end),0.05/50);
end

auxSig=find(ImportantBeta==1);

%plots convergence and qq plot
figure
    subplot(4,4,1)
    plot(betamat(auxSig(1),5000:50:end)')
    title('Coefficient 1')
    subplot(4,4,2)
    qqplot(betamat(auxSig(1),5000:50:end)')
    %title('Coefficient 1')
    subplot(4,4,3)
    plot(betamat(auxSig(2),5000:50:end)')
    title('Coefficient 8')
    subplot(4,4,4)
    qqplot(betamat(auxSig(2),5000:50:end)')
    %title('qqplot for Significant Coefficient 8')
    subplot(4,4,5)
    plot(betamat(auxSig(3),5000:50:end)')
    title('Coefficient 9')
    subplot(4,4,6)
    qqplot(betamat(auxSig(3),5000:50:end)')
    subplot(4,4,7)
    plot(betamat(auxSig(4),5000:50:end)')
    title('Coefficient 10')
    subplot(4,4,8)
    qqplot(betamat(auxSig(4),5000:50:end)')
    subplot(4,4,9)
    plot(betamat(auxSig(5),5000:50:end)')
    title('Coefficient 23')
    subplot(4,4,10)
    qqplot(betamat(auxSig(5),5000:50:end)')
    subplot(4,4,11)
    plot(betamat(auxSig(6),5000:50:end)')
    title('Coefficient 36')
    subplot(4,4,12)
    qqplot(betamat(auxSig(6),5000:50:end)')
    subplot(4,4,13)
    plot(betamat(auxSig(7),5000:50:end)')
    title('Coefficient 41')
    subplot(4,4,14)
    qqplot(betamat(auxSig(7),5000:50:end)')
    
    

%Projection of the Posterior Mean

Betahat=mean(betamat(:,5000:50:end),2);
stdhat=std(betamat(:,5000:50:end),1,2);
Betahat2=zeros(size(Betahat));
auxSig=find(ImportantBeta==1);
Betahat2(auxSig)=Betahat(auxSig);
Phat=COEFF0(:,1:K)*Betahat2;

%Projecting the mean into the image space
MeanAll=cell(4096,1);

for ppp=1:4096
   MeanAll{ppp}=zeros(6,6,6); 
end
for jj=1:npart1
    aux_mean=Phat((jj-1)*R+1:jj*R,1);
    MeanAll{index_partition2(jj)}=ktensor(Lamb{jj},AMat{jj},BMat{jj},CMat{jj},aux_mean');
end



ProjectioMean=zeros(96,96,96);
dimP=96;
dim2=96;

SP0=16;
SP1=16;
count=1;

for i=1:SP0
    for j=1:SP0
        for k=1:SP1
        ProjectionMean((dimP/SP0)*(i-1)+1:i*(dimP/SP0),(dimP/SP0)*(j-1)+1:j*(dimP/SP0),(dim2/SP1)*(k-1)+1:k*(dim2/SP1))=MeanAll{count};
        count=count+1;
        end
    end
end

ProjM=make_nii(double(abs(ProjectionMean)));
%view_nii(ProjM)
save_nii(ProjM,'PosteriorMean_Abs')

