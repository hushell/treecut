function [cl, Z, pcorr, acc] = test_lr(net, X, y, pPosR, pPos) 
% X: (nsamples,dim)
% y: (nsamples,1)

pcorr = 0;
nclass = 2;

Z = glmfwd(net, X);
if nargin == 5 % prior scaling
    posScal = pPos / pPosR;
    negScal = (1-pPos) / (1-pPosR+(pPosR==1));
    Z(:,1) = Z(:,1) * posScal;
    Z(:,2) = Z(:,2) * negScal;
    Z = bsxfun(@rdivide, Z, sum(Z,2));
end
[foo, cl] = max(Z');

if nargin < 2
    return
end

% recode y in one-of-nclass format
id = eye(nclass);
t = id(y,:);

ctot = sum(t);  % number of samples per class
cm = zeros(nclass); % confusion matrix

nsp = size(X,1);
for i = 1:nsp
    cm(y(i),cl(i)) = cm(y(i),cl(i)) +1;
end

pcorr = diag(cm) ./ ctot';
acc = trace(cm) ./ sum(cm(:));

labels = {'presence','absence'};
for i = 1:nclass
    fprintf('%15s %5.3f\n',labels{i}, pcorr(i));
end

