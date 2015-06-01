function logprob = poe_logprob(wws, x, siz, pis, sigs)
% modified from Yair Weiss's code

if nargin < 4
    pis = [0.1940 0.0906 0.4631 0.0423 0.1021 0.0816 0.0028 0.0234]';
    sigs = [0.0032 0.0147 0.0215 0.0316 0.0464 0.0681 0.1000 0.1468];
end

imsize=size(x);
nfilters = size(wws, 2);

pad_lo = ceil(0.5 * (siz - 1));
pad_hi = floor(0.5 * (siz - 1));
logprob=0;

for j = 1:nfilters
    % Filter mask 
    f  = reshape(wws(end:-1:1, j), siz);
    
    % Convolve and pad image appropriately
    tmp = zeros(size(x));
    tmp(1+pad_lo(1):end-pad_hi(1), 1+pad_lo(2):end-pad_hi(2)) = ...
        conv2(x, f, 'valid');
    
    logp=mog(tmp(:),pis,sigs);
    logprob=logprob+logp;
end

function logprob=mog(xx,pis,sigs)
  
xx=xx(:);
pis=pis(:);
sigs=sigs(:);
 
ss=1./sigs.^2;

err=exp(-0.5*xx.^2*(ss)');
err=err*diag(pis./sigs);
prob=sum(err,2);
 
logprob=sum(log(prob));
