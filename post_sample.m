function samples = post_sample(segTree, N)
%

numTotalNodes = segTree.numTotalNodes;
numLeafNodes = segTree.numLeafNodes;
samples = cell(1,N);

for n = 1:N
    unirand = rand(numTotalNodes,1);
    activeNodes = unirand < segTree.posterior;
    
    segLabels = zeros(numLeafNodes,1);
    for i = 1:numTotalNodes
        if activeNodes(i) == 1
            segLabels(segTree.leafsUnder{i}) = i;
        end
    end
    assert(all(segLabels > 0));
    samples{n} = segLabels;
end
