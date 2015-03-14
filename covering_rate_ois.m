function [cntR, sumR] = covering_rate_ois(seg, groundTruth)
%

[matches] = match_segmentations(seg, groundTruth);
matchesGT = max(matches, [], 1);
    
nsegs = numel(groundTruth);
regionsGT = [];
total_gt = 0;
for s = 1 : nsegs
    groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
    regionsTmp = regionprops(groundTruth{s}.Segmentation, 'Area');
    regionsGT = [regionsGT; regionsTmp];
    total_gt = total_gt + max(groundTruth{s}.Segmentation(:));
end

cntR = 0;
sumR = 0;
for r = 1 : numel(regionsGT),
    cntR = cntR + regionsGT(r).Area*matchesGT(r);
    sumR = sumR + regionsGT(r).Area;
end

