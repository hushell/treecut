function aloglik = average_loglik(Y)
% Y = [y1,...,yN]

D = size(Y,1);
mu = mean(Y,2);
sigma = cov(Y,1); % 1 means normalized by 1/N othewise by 1/N-1

loglik = -N/2 * (D*log(2*pi) + logdet(sigma) + D);
aloglik = loglik / N;


function y = logdet(A)
% Written by Tom Minka

U = chol(A);
y = 2*sum(log(diag(U)));
