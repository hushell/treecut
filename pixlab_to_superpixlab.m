function segLabels = pixlab_to_superpixlab(segMap, labMap)
%

numLeafNodes = sum(unique(segMap) > 0);
segLabels = zeros(numLeafNodes,1);

for i = 1:numLeafNodes
    labs = labMap(segMap == i);
    uniLab = unique(labs);
    assert(numel(uniLab) == 1);
    segLabels(i) = uniLab;
end