function [fval,grad] = func_single(thisTree, w, feats)
%

scal = 1e-3;

if nargin < 3
    fdim = 6+1;
    feats = zeros(thisTree.numTotalNodes,fdim); 
    dist_to_r = zeros(thisTree.numTotalNodes,1);
    for k = thisTree.numTotalNodes:-1:thisTree.numLeafNodes+1
        kids = thisTree.getKids(k);
        par = thisTree.getParent(k);
        if par == 0; ucm_par = 1; else ucm_par = thisTree.ucm(par); end
        dist_to_r(kids) = dist_to_r(k) + 1;
        feats(k,:) = [...
            ucm_par,thisTree.ucm(k),sort(thisTree.ucm(kids))', ...
            numel(thisTree.leafsUnder{k})/thisTree.numLeafNodes, ...
            dist_to_r(k), ...
            1
        ];
    end
end

p = ones(thisTree.numTotalNodes,1);
tmp = feats*w;
tmp = sigmf(tmp, [1,0]);

p(thisTree.numLeafNodes+1:thisTree.numTotalNodes) = tmp(thisTree.numLeafNodes+1:thisTree.numTotalNodes,1);
p(p == 0) = p(p == 0) + 0.001;
p(p == 1) = p(p == 1) - 0.001;

[aftTree,segLabels] = inference_temp(thisTree, p, scal);

% A+\A: 1  A-: 0  Leafs: 3  A: 2
govern = zeros(aftTree.numTotalNodes,1);
for k = aftTree.numTotalNodes:-1:aftTree.numLeafNodes+1
    kids = aftTree.getKids(k);
    if aftTree.activeNodes(k) == 1 || govern(k) == 1
        govern(kids) = 1; % all nodes below active nodes are govern, indicated as 1
    end
end
govern(1:aftTree.numLeafNodes) = 3;
govern(~govern & aftTree.activeNodes) = 2;

ind_1 = govern == 0; % A-
ind_2 = govern == 2; % A
feat_1 = feats(ind_1,:);
feat_2 = feats(ind_2,:);
prior_1 = p(ind_1);
prior_2 = p(ind_2);

grad = zeros(size(w));
grad = grad + feat_1*(-1+prior_1) + feat_2*prior_2;

q_plus = zeros(aftTree.numTotalNodes,1);
q_minus = zeros(aftTree.numTotalNodes,1);
for i = numTotalNodes:-1:1
    %kids = segTree.getKids(i);
    %il = kids(1); ir = kids(2);
    p_i = pp(i);

    pai_to_root = [];
    par = aftTree.pp(i);
    while par ~= 0
        pai_to_root(end+1) = par;
        par = aftTree.pp(par);
    end
    
    p_plus(i) = aftTree.posterior(i) * prod(1-aftTree.posterior(pai_to_root));  
    p_minus(i) = (1-aftTree.posterior(i)) * prod(1-aftTree.posterior(pai_to_root));  
end

grad = grad - feats*(p_plus.*(-1+prior) + p_minus.*prior);
