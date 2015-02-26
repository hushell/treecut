clear
close all

dataset = 'test';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/ucm_trees/' dataset '/'];
pt_dir   = ['./output/ucm_processed_trees/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

%tot = numel(all_files);
tot = 50;
for i = 2:tot
    % prepare data
    [~,name] = fileparts(all_files(i).name);
    load([tree_dir name '_tree.mat']); % tree
    load([ucm2_dir name '.mat']); % ucm2
    load([gt_dir name '.mat']); % gt
    img = imread([img_dir name '.jpg']); % img
    
    ucm = ucm2(3:2:end, 3:2:end);
    segMap = bwlabel(ucm <= 0, 4);
    
    nsegs = numel(groundTruth);
    for s = 1:nsegs
        groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
    end
    
    % test UCM
    el = strel('diamond',1);
    for j = 1:length(thres_arr)
        k = thres_arr(j);
        labMap = bwlabel(ucm <= k, 4);
        
        for m = 1:2
           tmp = imdilate(labMap,el);
           labMap(labMap == 0) = tmp(labMap == 0);
        end
        
        [PRI, VOI] = match_segmentations2(labMap, groundTruth);

        UCM_grid_PRI(j) = PRI;
        UCM_grid_VOI(j) = VOI;
        UCM_grid_nLab(j) = numel(unique(labMap));
    end % j
    [~,UCM_j] = max(UCM_grid_PRI);
    UCM_PRI = UCM_grid_PRI(UCM_j);
    UCM_VOI = UCM_grid_VOI(UCM_j);
    UCM_nLab = UCM_grid_nLab(UCM_j);
    g_thres = thres_arr(UCM_j);
    UCM_labMap = bwlabel(ucm <= g_thres, 4);
    for m = 1:2
        tmp = imdilate(UCM_labMap,el);
        UCM_labMap(UCM_labMap == 0) = tmp(UCM_labMap == 0);
    end
        
    % test treecut
    scal = 1e-3;
    n_r = 1;
    n_s = 50;
    log_ps = linspace(-2.5,-0.005,n_s); % exp(-2.5) ~ 0.08, exp(-0.1) = 0.9
    log_ps = [log_ps log(0.99) log(0.999) log(0.9999) log(0.99999) log(0.999999)];
    n_s = n_s + 5;
    
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
    
    for j = 1:n_s
        p = exp(log_ps(j));
        [aftTree,segLabels] = inference(thisTree, p, scal);
        
        [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
        TC_grid_PRI(j) = PRI;
        TC_grid_VOI(j) = VOI;
        TC_grid_nLab(j) = numel(unique(segLabels));
    end % j
    [~,TC_j] = max(TC_grid_PRI);
    TC_PRI = TC_grid_PRI(TC_j);
    TC_VOI = TC_grid_VOI(TC_j);
    TC_nLab = TC_grid_nLab(TC_j);
    g_p = exp(log_ps(TC_j));
    [aftTree,segLabels] = inference(thisTree, g_p, scal);
    [TC_PRI, TC_VOI, TC_labMap] = eval_seg(segMap, segLabels, groundTruth);


    fprintf('UCM: (%d,%d): PRI = %f, VOI = %f\n', i,UCM_j, UCM_PRI, UCM_VOI);
    fprintf('TC: (%d,%d): PRI = %f, VOI = %f\n', i,TC_j, TC_PRI, TC_VOI);
    figure(1000); aftTree.plotForest(g_thres);
    figure(1001); imagesc(UCM_labMap);
    figure(1002); imagesc(TC_labMap);
    pause
end % i 

