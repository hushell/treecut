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

thisTreePath = [pt_path 'train/100075_tree.mat'];
thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);

scal = 1e-3;
    n_r = 1;
    n_s = 50;
    grid_PRI = zeros(n_r,n_s);
    grid_VOI = zeros(n_r,n_s);
    grid_nLab = zeros(n_r,n_s);
    log_ps = linspace(-2.5,-0.005,n_s); % exp(-2.5) ~ 0.08, exp(-0.1) = 0.9
    log_ps = [log_ps log(0.99) log(0.999) log(0.9999) log(0.99999) log(0.999999)];
    n_s = n_s + 5;
    grid_segLabels = cell(n_r,n_s);
    %grid_labMap = cell(n_r,n_s);
    
    for j = 1:n_s
        p = exp(log_ps(j));
        [gtTree, gt_lliks, gt_labs] = best_gt_trees(segMap, groundTruth, thisTree, p, scal);
        [aftTree,segLabels] = inference(gtTree, p, scal);
        
        [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
        grid_PRI(j) = PRI;
        grid_VOI(j) = VOI;
        grid_nLab(j) = numel(unique(segLabels));
        
        grid_segLabels{j} = segLabels;
        %grid_labMap{j} = labMap;
        
        %fprintf('(%d,%d): PRI = %f, VOI = %f\n', i,j, PRI, VOI);
        %imagesc(labMap);
        %vis_seg(segMap, img, segLabels);
    end % j