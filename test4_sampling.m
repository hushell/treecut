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

% p = 0.9, scale = 1e-3
thisTreePath = [pt_path 'train/100075_tree.mat'];
thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
[aftTree,segLabels] = inference(thisTree, 0.99, 1e-3);
segLabels = shuffle_labels(segLabels);
%vis_seg(segMap, img, segLabels);
[PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
fprintf('PRI = %.2f, VOI = %.2f, nSeg = %d\n', PRI, VOI, numel(unique(segLabels)));
imagesc(labMap);

% samples
N = 100;
samples = post_sample(aftTree, N);
ave_img = zeros(size(segMap));
for n = 1:N
    [PRI, VOI, labMap] = eval_seg(segMap, samples{n}, groundTruth);
    %fprintf('PRI = %.2f, VOI = %.2f, nSeg = %d\n', PRI, VOI, numel(unique(samples{n})));
    %figure; imagesc(labMap);
    %figure; vis_seg(segMap, img, samples{n});
    bdry = seg2bdry(labMap, 'imageSize');
    ave_img = ave_img + bdry;
end

ave_img = ave_img / N;
