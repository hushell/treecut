clear
close all

pt_path = 'output/processed_trees/';

% prepare data
load output/trees/bear_legs2_tree.mat % tree
img = imread('data/bear_legs2.png');
load data/bear_legs2_ucm2.mat % ucm2
ucm = ucm2(3:2:end, 3:2:end);
segMap = bwlabel(ucm <= 0, 4);
load data/bear_legs2_gt.mat % gt
nsegs = numel(groundTruth);
for s = 1:nsegs
    groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
end

% p = 0.9, scale = 0.9
thisTreePath = [pt_path 'bear_legs2_tree.mat'];
thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
[aftTree,segLabels] = inference(thisTree, 0.6, 1e-3);
segLabels = shuffle_labels(segLabels);
%vis_seg(segMap, img, segLabels);
[PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
fprintf('PRI = %.2f, VOI = %.2f, nSeg = %d\n', PRI, VOI, numel(unique(segLabels)));
imagesc(labMap);


% samples
N = 10000;
samples = post_sample2(aftTree, N);
activeDist = sum(samples,1) ./ N; activeDist = activeDist';
