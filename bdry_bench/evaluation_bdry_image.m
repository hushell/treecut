function [thresh,cntR,sumR,cntP,sumP] = evaluation_bdry_image(pb, groundTruth, prFile, nthresh, maxDist, thinpb)
% [thresh,cntR,sumR,cntP,sumP] = boundaryPR_image(inFile,gtFile, prFile, nthresh, maxDist, thinpb)
%
% Calculate precision/recall curve.
%
% INPUT
%   prFile  : Temporary output for this image.
%	nthresh	: Number of points in PR curve.
%   MaxDist : For computing Precision / Recall.
%   thinpb  : option to apply morphological thinning on segmentation
%             boundaries.
%
% OUTPUT
%	thresh		Vector of threshold values.
%	cntR,sumR	Ratio gives recall.
%	cntP,sumP	Ratio gives precision.
%
if nargin<6, thinpb = 1; end
if nargin<5, maxDist = 0.0075; end
if nargin<4, nthresh = 99; end

thresh = linspace(1/(nthresh+1),1-1/(nthresh+1),nthresh)';

% zero all counts
cntR = zeros(size(thresh));
sumR = zeros(size(thresh));
cntP = zeros(size(thresh));
sumP = zeros(size(thresh));

for t = 1:nthresh,
    bmap = (pb>=thresh(t));
    
    % thin the thresholded pb to make sure boundaries are standard thickness
    if thinpb,
        bmap = double(bwmorph(bmap, 'thin', inf));    % OJO
    end
    
    % accumulate machine matches, since the machine pixels are
    % allowed to match with any segmentation
    accP = zeros(size(bmap));
    
    % compare to each seg in turn
    for i = 1:numel(groundTruth),
        % compute the correspondence
        [match1,match2] = correspondPixels(bmap, double(groundTruth{i}.Boundaries), maxDist);
        % accumulate machine matches
        accP = accP | match1;
        % compute recall
        sumR(t) = sumR(t) + sum(groundTruth{i}.Boundaries(:));
        cntR(t) = cntR(t) + sum(match2(:)>0);
    end
    
    % compute precision
    sumP(t) = sumP(t) + sum(bmap(:));
    cntP(t) = cntP(t) + sum(accP(:));

end

% output
fid = fopen(prFile,'w');
if fid==-1,
    error('Could not open file %s for writing.', prFile);
end
fprintf(fid,'%10g %10g %10g %10g %10g\n',[thresh cntR sumR cntP sumP]');
fclose(fid);

