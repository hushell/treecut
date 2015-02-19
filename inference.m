function [segTree, segLabels] = inference(segTree, p, scal, verbose)
%

if nargin < 4
    verbose = 0;
end

numLeafNodes = segTree.numLeafNodes;
numTotalNodes = segTree.numTotalNodes;
llik = segTree.llik * scal;

%% bottom-up phase: E(i) = P(Y_i) 
% leaves
segTree.E = zeros(numTotalNodes,1);
segTree.M = zeros(numTotalNodes,1);
segTree.posterior = zeros(numTotalNodes,1);
segTree.v = ones(numTotalNodes,1); % 1--govern 2--cut

segTree.E(1:numLeafNodes) = llik(1:numLeafNodes);
segTree.M(1:numLeafNodes) = llik(1:numLeafNodes);
segTree.posterior(1:numLeafNodes) = 1;

pp = zeros(numTotalNodes,1);
pp(:) = p; % global p(v_i)

% E, M, posterior, v 
for i = numLeafNodes+1:numTotalNodes
    kids = segTree.getKids(i);
    il = kids(1); ir = kids(2);
    p_i = pp(i);
    L_i = llik(i);
    
    % Ei = log( exp(log_pi + Li) + exp(log(1-pi) + Eil + Eir) )
    E_il = segTree.E(il);
    E_ir = segTree.E(ir);
    PD = log(p_i) + L_i; 
    PI = log(1-p_i) + E_il + E_ir;
    maxP = max(PD, PI);
    E_i = log( exp(PD-maxP) + exp(PI-maxP) ) + maxP;
    post = exp(PD - E_i);
    segTree.E(i) = E_i;
    segTree.posterior(i) = post;
    %fprintf('node %d (sum): post = %f, PD = %e, PI = %e\n', i, post, PD, PI);

    % Mi = max(log_pi + Li, log(1-pi) + Mil + Mir)
    %M_il = segTree.M(il) * scal;
    %M_ir = segTree.M(ir) * scal;
    M_il = segTree.M(il);
    M_ir = segTree.M(ir);
    PD_m = log(p_i) + L_i; 
    PI_m = log(1-p_i) + M_il + M_ir;
    [segTree.M(i),v_i] = max([PD_m, PI_m]);
    segTree.v(i) = v_i;
    if verbose
        fprintf('node %d (max): v_i = %d, PD* = %e, PI* = %e, diff = %e\n', i, v_i, PD_m, PI_m, PD_m-PI_m);
    end
end

%% top-down backtracking
global activeNodes
activeNodes = zeros(numTotalNodes,1); 
backtrack(segTree, numTotalNodes);
segTree.activeNodes = activeNodes;

segLabels = zeros(numLeafNodes,1);
for i = 1:numTotalNodes
    if activeNodes(i) == 1
        segLabels(segTree.leafsUnder{i}) = i;
    end
end
assert(all(segLabels > 0));


function backtrack(segTree, curr)
%

global activeNodes

if segTree.v(curr) == 1 % NOTE v(leaf) = 1
    activeNodes(curr) = 1;
    return
end

kids = segTree.getKids(curr);
il = kids(1); ir = kids(2);
assert(il > 0 && ir > 0);

% pre-order traversal
backtrack(segTree, il);
backtrack(segTree, ir);
