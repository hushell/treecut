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

all_grid_segLabels = cell(1,numel(all_files));
all_grid_PRI = cell(1,numel(all_files));
all_grid_VOI = cell(1,numel(all_files));
all_grid_nLab = cell(1,numel(all_files));
all_grid_R = cell(1,numel(all_files));

ave_PRI = 0;
ave_VOI = 0;
cntR_best = 0;
sumR_best = 0;

for i = 1:numel(all_files)
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
    
    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
    
    % test
    scal = 1e-3;
    p = 1-thisTree.ucm;
    [aftTree,segLabels] = inference_temp(thisTree, p, scal);
        
    [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
    [cntR, sumR] = covering_rate_ois(labMap, groundTruth);
        
    all_grid_segLabels{i} = segLabels;
    all_grid_PRI{i} = PRI;
    all_grid_VOI{i} = VOI;
    all_grid_nLab{i} = numel(unique(segLabels));
    ave_PRI = ave_PRI + PRI;
    ave_VOI = ave_VOI + VOI;
    
    R = cntR ./ (sumR + (sumR==0));
    all_grid_R{i} = R;
    cntR_best = cntR_best + cntR;
    sumR_best = sumR_best + sumR;
    
    fprintf('%d: PRI = %f, VOI = %f, COV = %f\n', i, PRI, VOI, R);
    %imagesc(labMap);
    %vis_seg(segMap, img, segLabels);
end % i 

ave_PRI = ave_PRI / numel(all_files);
ave_VOI = ave_VOI / numel(all_files);
R_best = cntR_best ./ (sumR_best + (sumR_best==0));
fprintf('*** ave_PRI = %f, ave_VOI = %f, ave_COV = %f\n', ave_PRI, ave_VOI, R_best);

save('BSDS_test9.mat');
