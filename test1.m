clear
close all

pt_path = 'output/processed_trees/';

% prepare data
load data/trees/100075_tree.mat % tree
img = imread('data/images/100075.jpg');
load data/ucm2/100075.mat % ucm2
ucm = ucm2(3:2:end, 3:2:end);
segMap = bwlabel(ucm <= 0, 4);
load data/groundTruth/100075.mat % gt
nsegs = numel(groundTruth);
for s = 1:nsegs
    groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
end

% p = 0.9, scale = 0.9
thisTreePath = [pt_path 'train/100075_tree.mat'];
thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
[aftTree,segLabels] = inference(thisTree, 0.0450, 1e-4);
segLabels = shuffle_labels(segLabels);
%vis_seg(segMap, img, segLabels);
[PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
fprintf('PRI = %f, VOI = %f\n', PRI, VOI);
imagesc(labMap);

return 

% samples
N = 6;
samples = post_sample(aftTree, N);
for n = 1:N
    figure; vis_seg(segMap, img, samples{n});
end
close all
 
% check different p's
%pps = [0.001, 0.01, 0.1, 0.9, 0.99, 0.999];
%pps = [0.999 0.9999 0.999999];
pps = 0.1:0.1:0.8;
for p = pps
    [aftTree,segLabels] = inference(thisTree, img, segMap, p, 0.9);
    [PRI, VOI] = eval_seg(segMap, segLabels, groundTruth);
    fprintf('*** p = %f, PRI = %f, VOI = %f\n', p, PRI, VOI);
    figure(int32(p*10000)); vis_seg(segMap, img, segLabels); 
    pause
end
