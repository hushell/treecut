function [fval,grad] = func_single(w, alg_params, feats)
%

thisTree = alg_params.thisTree;
scal = alg_params.scal;
gt_sub = alg_params.gt;
segMap = alg_params.segMap;

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

% inference
p = ones(thisTree.numTotalNodes,1);
tmp = feats*w;
tmp = sigmf(tmp, [1,0]);

p(thisTree.numLeafNodes+1:thisTree.numTotalNodes) = tmp(thisTree.numLeafNodes+1:thisTree.numTotalNodes,1);
p(p <= 0) = p(p <= 0) + 0.001;
p(p >= 1) = p(p >= 1) - 0.001;

[aftTree,segLabels] = inference(thisTree, p, scal);

if alg_params.do_eval == 1
    % eval
    numSegs = length(segLabels);
    labMap = zeros(size(segMap));
    for l = 1:numSegs
        labMap(segMap == l) = segLabels(l);
    end
    el = strel('diamond',1);
    for m = 1:2
       tmp = imdilate(labMap,el);
       labMap(labMap == 0) = tmp(labMap == 0);
    end

    [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
    COV = cntR ./ (sumR + (sumR==0));
    fprintf('COV = %f, nLabs = %d\n', COV, numel(unique(segLabels)));
end

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

% gradient
ind_0 = govern == 0; % A-
ind_2 = govern == 2; % A
feat_0 = feats(ind_0,:);
feat_2 = feats(ind_2,:);
prior_0 = p(ind_0);
prior_2 = p(ind_2);

grad = zeros(size(w));
grad = grad + feat_2'*(-1+prior_2) + feat_0'*prior_0;

q_plus = zeros(aftTree.numTotalNodes,1);
q_minus = zeros(aftTree.numTotalNodes,1);
for i = aftTree.numTotalNodes:-1:1
    pai_to_root = [];
    par = aftTree.pp(i);
    while par ~= 0
        pai_to_root(end+1) = par;
        par = aftTree.pp(par);
    end
    
    post_abov = 1-aftTree.posterior(pai_to_root); 
    if isempty(post_abov); post_abov = 1; end
    q_plus(i) = aftTree.posterior(i) * prod(post_abov);  
    q_minus(i) = (1-aftTree.posterior(i)) * prod(post_abov);  
end

grad = grad - feats'*(q_plus.*(-1+p) + q_minus.*p);
grad = -grad;

% fval wrt current w
fval = sum(aftTree.llik(ind_2)) + sum(log(prior_2)) + sum(log(1-prior_0)) - aftTree.E(end);
fval = -fval;
fprintf('fval = %f\n', -fval);
