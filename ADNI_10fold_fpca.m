% This code runs the FPCA for the real data as described in Section 4.1 of
% the paper

% TPRM:  Tensor partition regression models with applications in  
% imaging biomarker detection (Michelle F. Miranda, Hongtu Zhu and Joseph
% G. Ibrahim

%Load the data, the data is in 4 different files due to memory constrains

load('Tensor_ADNI_4_red.mat')
load('Tensor_ADNI_3_red.mat')
load('Tensor_ADNI_2_red.mat')
load('Tensor_ADNI_1_red.mat')

n1=100;
n2=81;
n3=100;
n4=121;

YTotal=zeros(96,96,96,402);
YTotal(:,:,:,1:100)=Tensor_ADNI_1;
YTotal(:,:,:,101:181)=Tensor_ADNI_2;
YTotal(:,:,:,182:281)=Tensor_ADNI_3;
YTotal(:,:,:,282:n1+n2+n3+n4)=Tensor_ADNI_4;

clear Tensor_ADNI_1 Tensor_ADNI_2 Tensor_ADNI_3 Tensor_ADNI_4

XR1=zeros(402,884736); %96*96*96=884736

for i=1:402
    XR1(i,:)=reshape(YTotal(:,:,:,i),1,884736);
end

clear YTotal

load('C:\Users\mfmiranda\Documents\MATLAB\AOAS_Revision\ADNI_10fold_sets');
Y=[ones(181,1); zeros(221,1)]; 
PAccuracy_fpcalasso=zeros(10,1);
PAccuracy_tree=zeros(10,1);
PAccuracy_svm=zeros(10,1);
for sim=1:10;
tic
indexTra=training(CVP,sim);
indexTest=test(CVP,sim);

Y_train=Y(indexTra,1);
Y_test=Y(indexTest,1);

N_test=size(Y_test,1);
N_train=size(Y_train,1);

X1_train=XR1(indexTra,:);
X1_test=XR1(indexTest,:);

[COEFF0,G0,latent0]=princomp(X1_train,'econ');
%Take K that explain 99% of variability
scree_train=[0;cumsum(latent0)./sum(latent0)];
idx=0:size(latent0,1);
K=min(idx(scree_train>=0.99)); 

%Regularized lasso with logistic
G0_train=G0(:,1:K);
[B,FitInfo] = lassoglm(G0_train,Y_train,'binomial');
G0_test=X1_test*COEFF0(:,1:K);
mindev=min(FitInfo.Deviance);
index_dev=find(FitInfo.Deviance==mindev);
aux_pred=exp(G0_test*B(:,index_dev));
Predicted_mu=aux_pred./(1+aux_pred);
PredictedY=Predicted_mu>0.5;

PAccuracy_fpcalasso(sim,1)=sum(PredictedY==Y_test)/N_test;


% tree
Tree = ClassificationTree.fit(G0_train,Y_train);
PAccuracy_tree(sim,1)=sum(predict(Tree,G0_test)==Y_test)/N_test;
% svm
Mdl = fitcsvm(G0_train,Y_train);
PAccuracy_svm(sim,1)=sum(predict(Mdl,G0_test)==Y_test)/N_test;


toc
end
save('ADNI_10fold_MethodsComparison','PAccuracy_tree','PAccuracy_fpcalasso','PAccuracy_svm')