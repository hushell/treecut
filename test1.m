clear
close all

pt_path = 'output/processed_trees/';

% prepare data
load output/trees/train/100075_tree.mat % tree
img = imread('data/images/train/100075.jpg');
load data/ucm2/train/100075.mat % ucm2
ucm = ucm2(3:2:end, 3:2:end);
segMap = bwlabel(ucm <= 0, 4);
load data/groundTruth/train/100075.mat % gt
nsegs = numel(groundTruth);
for s = 1:nsegs
    groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
end

% p = 0.9, scale = 0.9
thisTreePath = [pt_path 'train/100075_tree.mat'];
thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
[aftTree,segLabels] = inference(thisTree, 0.999, 1e-3);
segLabels = shuffle_labels(segLabels);
%vis_seg(segMap, img, segLabels);
[PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
[cntR, sumR, regInd, covRate] = covering_rate_ois(labMap, groundTruth);
R = cntR ./ (sumR + (sumR==0));
fprintf('PRI = %.2f, VOI = %.2f, COV = %.2f, nSeg = %d, covRate = %f\n', PRI, VOI, R, numel(unique(segLabels)), covRate);
imagesc(labMap);

figure; imagesc(ismember(labMap, regInd));
