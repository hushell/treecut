function scratch = vis_seg2(labMap, img)
% TODO: use ismember(segMap, [seg1, seg2, ...])

labsPool = unique(labMap(:));
numLabs = numel(labsPool);
colMap = hsv(numLabs);
scratch = img;

for i = 1:numLabs
    lab = labsPool(i);
    lInd = labMap == lab;
    s = scratch(:,:,1);
    s(lInd) = median(s(lInd));
    scratch(:,:,1) = s;
    s = scratch(:,:,2);
    s(lInd) = median(s(lInd));
    scratch(:,:,2) = s;
    s = scratch(:,:,3);
    s(lInd) = median(s(lInd));
    scratch(:,:,3) = s;
end

if nargout < 1
    imshow(scratch);
end
