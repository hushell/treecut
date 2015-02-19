function [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth)
%

numSegs = length(segLabels);
labMap = zeros(size(segMap));

for i = 1:numSegs
    labMap(segMap == i) = segLabels(i);
end

%labMap = padarray(labMap, [1,1]);
%filling in edge pixels
el = strel('diamond',1);
for i = 1:2
   tmp = imdilate(labMap,el);
   labMap(labMap == 0) = tmp(labMap == 0);
end

[PRI, VOI] = match_segmentations2(labMap, groundTruth);