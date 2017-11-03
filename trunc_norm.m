function x=trunc_norm(meanT,flagT)
no_gau=normrnd(meanT,1);
if flagT==0
    x=min(no_gau,0);
else
    x=max(no_gau,0);
end


