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

thisTreePath = [pt_path 'train/100075_tree.mat'];
thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);

scals = [1e-3 1e-4];
n_r = length(scals);
n_s = 100;
grid_PRI = zeros(n_r,n_s);
grid_VOI = zeros(n_r,n_s);
grid_nLab = zeros(n_r,n_s);
log_ps = linspace(-10,-0.1,n_s);

for i = 1:n_r
    for j = 1:n_s
        p = exp(log_ps(j));
        [aftTree,segLabels] = inference(thisTree, p, scals(i));
        %vis_seg(segMap, img, segLabels);
        [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
        grid_PRI(i,j) = PRI;
        grid_VOI(i,j) = VOI;
        grid_nLab(i,j) = numel(unique(segLabels));
        fprintf('(%d,%d): PRI = %f, VOI = %f\n', i,j, PRI, VOI);
        %imagesc(labMap);
    end
end
