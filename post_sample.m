function [samples, sample_actives] = post_sample(segTree, N)

numTotalNodes = segTree.numTotalNodes;
numLeafNodes = segTree.numLeafNodes;
samples = zeros(N,numLeafNodes);
sample_actives = zeros(N, numTotalNodes);

global activeNodes localActiveNodes

for n = 1:N
	activeNodes = zeros(1, numTotalNodes);
    unirand = rand(1, numTotalNodes);
    localActiveNodes = unirand <= segTree.posterior';

	backtrack(segTree, numTotalNodes);
    sample_actives(n,:) = activeNodes;

    segLabels = 1:numLeafNodes;
    for i = numLeafNodes+1:numTotalNodes
        if activeNodes(i) == 1
            segLabels(segTree.leafsUnder{i}) = i;
        end
    end
    %assert(all(segLabels > 0));
    samples(n,:) = segLabels;
end


function backtrack(segTree, curr)
%

global activeNodes localActiveNodes

if localActiveNodes(curr) == 1 
    activeNodes(curr) = 1;
    return
end

kids = segTree.getKids(curr);
il = kids(1); ir = kids(2);
%assert(il > 0 && ir > 0);
if ~any(kids)
    return
end

% pre-order traversal
backtrack(segTree, il);
backtrack(segTree, ir);
