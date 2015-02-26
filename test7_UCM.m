clear
close all

dataset = 'test';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/ucm_trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

all_grid_segLabels = cell(1,numel(all_files));
all_grid_PRI = cell(1,numel(all_files));
all_grid_VOI = cell(1,numel(all_files));
all_grid_nLab = cell(1,numel(all_files));

all_best_j_PRI = zeros(1,numel(all_files));
all_best_j_VOI = zeros(1,numel(all_files));

ave_PRI = 0;
ave_VOI = 0;

fp = fopen('log_test7.txt', 'w'); % ***

%tot = numel(all_files);
tot = 50;
for i = 1:tot
    % prepare data
    [~,name] = fileparts(all_files(i).name);
    load([tree_dir name '_tree.mat']); % tree
    load([ucm2_dir name '.mat']); % ucm2
    load([gt_dir name '.mat']); % gt
    img = imread([img_dir name '.jpg']); % img
    
    ucm = ucm2(3:2:end, 3:2:end);
    %segMap = bwlabel(ucm <= 0, 4);
    
    nsegs = numel(groundTruth);
    for s = 1:nsegs
        groundTruth{s}.Segmentation = double(groundTruth{s}.Segmentation);
    end
    
    % test
    el = strel('diamond',1);
    for j = 1:length(thres_arr)
        k = thres_arr(j);
        labMap = bwlabel(ucm <= k, 4);
        
        %[PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
        for i = 1:2
           tmp = imdilate(labMap,el);
           labMap(labMap == 0) = tmp(labMap == 0);
        end
        
        [PRI, VOI] = match_segmentations2(labMap, groundTruth);

        grid_PRI(j) = PRI;
        grid_VOI(j) = VOI;
        grid_nLab(j) = numel(unique(labMap));
        
    end % j
        
    all_grid_PRI{i} = grid_PRI;
    all_grid_VOI{i} = grid_VOI;
    all_grid_nLab{i} = grid_nLab;
    
    [~,best_j] = max(grid_PRI);
    all_best_j_PRI(i) = best_j;
    ave_PRI = ave_PRI + grid_PRI(best_j);
    
    [~,best_k] = min(grid_VOI);
    all_best_j_VOI(i) = best_k;
    ave_VOI = ave_VOI + grid_VOI(best_k);
    
    fprintf('img %d: best_PRI&VOI = (%f,%f), best_VOI = %f\n', i, grid_PRI(best_j), grid_VOI(best_j), grid_VOI(best_k));
    fprintf(fp, 'img %d: best_PRI&VOI = (%f,%f), best_VOI = %f\n', i, grid_PRI(best_j), grid_VOI(best_j), grid_VOI(best_k));
end % i 

ave_PRI = ave_PRI / tot;
ave_VOI = ave_VOI / tot;
fprintf('*** ave_PRI = %f, ave_VOI = %f\n', ave_PRI, ave_VOI);
fprintf(fp, '*** ave_PRI = %f, ave_VOI = %f\n', ave_PRI, ave_VOI);

fclose(fp);

save('BSDS_test7.mat'); % ***
