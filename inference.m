function [segTree, segLabels] = inference(segTree, img, segMap, p, scal)
%

numLeafNodes = sum(unique(segMap) > 0);
numTotalNodes = size(segTree.kids,1);
segTree.numLeafNodes = numLeafNodes;
segTree.numTotalNodes = numTotalNodes;
img = im2double(img);
img1 = img(:,:,1);
img2 = img(:,:,2);
img3 = img(:,:,3);

%% leafs of each subtree: leafsUnder
numLeafsUnder = ones(numTotalNodes,1);
leafsUnder = cell(numTotalNodes,1);
% leafs
for s = 1:numLeafNodes
    leafsUnder{s} = s;
    segTree.leafsUnder{s} = leafsUnder{s};
end
% internals
for n = numLeafNodes+1:numTotalNodes
    kids = segTree.getKids(n);
    numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
    leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
end
segTree.leafsUnder = leafsUnder;

%% aloglik of each node
segTree.allik = zeros(numTotalNodes,1);
segTree.llik = zeros(numTotalNodes,1);
for i = 1:numTotalNodes
    pidx = ismember(segMap, leafsUnder{i});
    rgb = double([img1(pidx), img2(pidx), img3(pidx)]); % m x 3
    [allik,ll] = gaussian_loglik(rgb);
    segTree.allik(i) = allik;
    segTree.llik(i) = ll;
    fprintf('node %d: allik = %e, ll = %e\n', i, allik, ll);
end
fprintf('--------------\n');

%% bottom-up phase: E(i) = P(Y_i) 
% leaves
segTree.E = zeros(numTotalNodes,1);
segTree.M = zeros(numTotalNodes,1);
segTree.posterior = zeros(numTotalNodes,1);
segTree.v = ones(numTotalNodes,1); % 1--govern 2--cut

%segTree.E(1:numLeafNodes) = exp(segTree.llik(1:numLeafNodes));
%segTree.M(1:numLeafNodes) = exp(segTree.llik(1:numLeafNodes));
segTree.E(1:numLeafNodes) = segTree.llik(1:numLeafNodes);
segTree.M(1:numLeafNodes) = segTree.llik(1:numLeafNodes);
segTree.posterior(1:numLeafNodes) = 1;

pp = zeros(numTotalNodes,1);
pp(:) = p; % global p(v_i)

% 
for i = numLeafNodes+1:numTotalNodes
    kids = segTree.getKids(i);
    il = kids(1); ir = kids(2);
    L_i = segTree.llik(i) * scal;
    
    % E
    %E_il = segTree.E(il);
    %E_ir = segTree.E(ir);
    %segTree.E(i) = p * exp(L_i) + (1-p) * E_il * E_ir;

    % M
    M_il = segTree.M(il) * scal;
    M_ir = segTree.M(ir) * scal;
    %[segTree.M(i),v_i] = max([p * exp(L_i), (1-p) * M_il * M_ir]);
    PD = log(p) + L_i; 
    PI = log(1-p) + M_il + M_ir;
    [segTree.M(i),v_i] = max([PD, PI]);
    segTree.v(i) = v_i;
    segTree.posterior(i) = PD / (PD + PI);
    fprintf('node %d: v_i = %d, PD = %e, PI = %e\n', i, v_i, PD, PI);
end

%% top-down backtracking
global activeNodes
activeNodes = zeros(numTotalNodes,1); 
backtrack(segTree, numTotalNodes);
segTree.activeNodes = activeNodes;

segLabels = zeros(numLeafNodes,1);
for i = 1:numTotalNodes
    if activeNodes(i) == 1
        segLabels(leafsUnder{i}) = i;
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
