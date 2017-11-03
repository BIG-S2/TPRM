%This function samples delta, the probability for the mixture model 
%for each j

function [deltaj, betaj, pi_delta]=sample_deltaj(X,ydraw,pi_delta,betadraw,KK)
deltaj=zeros(KK,1);
betaj=zeros(KK,1);
sigma2=1000;
tau=0.0001;
alpha_pi=0.5;
beta_pi=0.5;
ydrawtilda=ydraw-X*betadraw;
for j=1:KK
aux1=sum(X(:,j).^2);
wi=ydrawtilda+betadraw(j,1)*X(:,j); %this is wi_tilda in the paper
aux2=sum(wi.*X(:,j)); 
p1_star=-0.5*(1/sigma2)*(betadraw(j,1))^2+log(pi_delta);
p0_star=-0.5*(1/tau)*(betadraw(j,1))^2+log(1-pi_delta);
M=max(p1_star,p0_star);
p1_tilda=exp(p1_star-M);
p0_tilda=exp(p0_star-M);
deltaj(j,1)=binornd(1,p1_tilda/(p0_tilda+p1_tilda));
%Sampling betaj
if deltaj(j,1)==0
    betaj(j,1)=normrnd(aux2/(aux1+1/tau),1/sqrt((aux1+1/tau)));
else
    betaj(j,1)=normrnd(aux2/(aux1+1/sigma2),1/sqrt((aux1+1/sigma2)));
end

end

pi_delta=betarnd(alpha_pi+sum(deltaj),beta_pi+KK-sum(deltaj));