function ws = poe_learn(xs, block, vis)
% modified from Yair Weiss's code
if nargin < 3
    vis = 0;
end

Kwhiten=whiteningFilter(xs,block);
basis=makeBasis(Kwhiten,2);

%load data/derivPisSigs; % one set of GSM potentials
pis = [0.1940 0.0906 0.4631 0.0423 0.1021 0.0816 0.0028 0.0234]';
sigs = [0.0032 0.0147 0.0215 0.0316 0.0464 0.0681 0.1000 0.1468];
%load data/TdistPisSigs; % another set of GSM potentials 
 
[nxs,~] = size(xs);
if nxs >= 5000
    xs=xs(1:10:end,:); % for fast training
end

[ws,orthVect]=basisRotation(xs,pis,sigs,basis);

if vis == 1
    figure; showFilts(ws,block);
end

function [Kwhiten,A]=whiteningFilter(xs,siz)
% calculate the zero phase whitening filter using PCA
% xs is a npatches*npixels _in_patch matrix where each row is a patch
% siz is the size of the patch (e.g. [5 5])
cov=xs'*xs;
[uu,ss,vv]=svd(cov);
dd=diag(ss);
D=diag(sqrt(1./dd));
A=uu*D*uu';
Kwhiten=reshape(A(round(prod(siz)/2),:),siz);
Kwhiten=Kwhiten';
Kwhiten=Kwhiten-mean(Kwhiten(:));
Kwhiten=Kwhiten/norm(Kwhiten(:));

function [ws]=makeBasis(Kwhiten,shift)
ws=[];
bigK=Kwhiten;
    
for dx=[-shift:shift]
    for dy=[-shift:shift]
        zeroIm=zeros(2*shift+1);
        zeroIm(shift+1+dx,shift+1+dy)=1;
        newK=conv2(bigK,zeroIm,'same');
        ws=[ws newK(:)];
    end
end

