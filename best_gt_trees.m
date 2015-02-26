function [gtTree, gt_lliks, gt_labs] = best_gt_trees(seg, groundTruth, segTree, p, scal)
%

numLeafNodes = segTree.numLeafNodes;
numTotalNodes = segTree.numTotalNodes;
nGT = numel(groundTruth);
gt_lliks = cell(1, nGT);
gt_labs = cell(1, nGT);

assert(numel(unique(seg(seg>0))) == numLeafNodes);

ave_llik = zeros(numTotalNodes,1);
gtTree = segTree;
llik = zeros(numTotalNodes,1);

for i = 1:nGT
    gt_seg = groundTruth{i}.Segmentation;
    seg_cnts = seg_gt_counts(seg, gt_seg);
    
    for n = 1:numTotalNodes
        llik(n) = purity_lik(seg_cnts, gtTree.leafsUnder{n});
    end
    
    gtTree.llik = llik;
    %[gtTree, gtLabels] = inference(gtTree, p, scal);
    
    gt_lliks{i} = llik;
    %gt_labs{i} = gtLabels;
    
    ave_llik = ave_llik + llik;
end

ave_llik = ave_llik / nGT;
gtTree.llik = ave_llik;


%% helpers
function pl = purity_lik(seg_cnts, leafs)

cnts = sum(seg_cnts(leafs,:),1);
tot = sum(cnts);
cnts = cnts ./ tot;
pl = tot * dot(cnts, log(cnts+1e-20));


function seg_cnts = seg_gt_counts(seg, gt)

numSegs = numel(unique(seg(seg>0)));
numGts = numel(unique(gt(:)));

assert(numSegs == max(seg(:)));
assert(numGts == max(gt(:)));

% segTotal = regionprops(seg, 'Area');
% segTotal = [segTotal.Area];
% gtTotal = regionprops(gt, 'Area');
% gtTotal = [gtTotal.Area];

seg_cnts = zeros(numSegs, numGts);

for i = 1:numSegs
    lab_invol = gt(seg == i);
    lab_occur = unique(lab_invol);
    lab_cnts = histc(lab_invol, lab_occur);
        
    seg_cnts(i,lab_occur) = lab_cnts; 
end

