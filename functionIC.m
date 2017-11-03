function IC= functionIC(data,alpha)
% Inputs - data: a vector of any size
%          alpha: 0<alpha<1 indicating the level of significance

% Output - IC: 0: the 1-alpha credible interval contains 0
%              1: the 1-alpha credible interval doesn't contains 0

teste=quantile(data,[alpha/2 1-(alpha/2)]);
low=teste(1,1);
hi=teste(1,2);
r = 0>=low & 0<=hi;
IC=1-r;