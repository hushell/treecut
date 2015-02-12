function vis_seg(segMap, img, segLabels)
% TODO: use ismember(segMap, [seg1, seg2, ...])

numSegs = length(segLabels);
labsPool = unique(segLabels);
numLabs = numel(labsPool);
colMap = hsv(numLabs);
scratch = img;

for i = 1:numSegs
    col = colMap(segLabels(i) == labsPool,:);
    s = scratch(:,:,1);
    s(segMap == i) = s(segMap == i)/3 + 100*col(1);
    scratch(:,:,1) = s;
    s = scratch(:,:,2);
    s(segMap == i) = s(segMap == i)/3 + 100*col(2);
    scratch(:,:,2) = s;
    s = scratch(:,:,3);
    s(segMap == i) = s(segMap == i)/3 + 100*col(3);
    scratch(:,:,3) = s;
end

imshow(scratch);
