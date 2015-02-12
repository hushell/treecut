function [aloglik,loglik] = gaussian_loglik(Y)
% Y = [y1;...;yN]

[N,D] = size(Y);
mu = mean(Y,1);
sigma = cov(Y,1); % 1 means normalized by 1/N othewise by 1/N-1

loglik = -N/2 * (D*log(2*pi) + logdet(sigma) + D);
aloglik = loglik / N;


function y = logdet(A)
% Written by Tom Minka

% if any(eig(A) <= 0)
%     y = log(det(A));
%     return
% end
% 
% U = chol(A);
% y = 2*sum(log(diag(U)));
y = log(det(A));
