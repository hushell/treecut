clear
close all

dataset = 'test';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

all_grid_segLabels = cell(1,numel(all_files));
%all_grid_labMap = cell(1,numel(all_files));
all_grid_PRI = cell(1,numel(all_files));
all_grid_VOI = cell(1,numel(all_files));
all_grid_nLab = cell(1,numel(all_files));

all_best_j_PRI = zeros(1,numel(all_files));
all_best_j_VOI = zeros(1,numel(all_files));

ave_PRI = 0;
ave_VOI = 0;

fp = fopen('log_test6.txt', 'w');

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
        
    all_grid_segLabels{i} = grid_segLabels;
    %all_grid_labMap{i} = grid_labMap;
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

ave_PRI = ave_PRI / numel(all_files);
ave_VOI = ave_VOI / numel(all_files);
fprintf('*** ave_PRI = %f, ave_VOI = %f\n', ave_PRI, ave_VOI);
fprintf(fp, '*** ave_PRI = %f, ave_VOI = %f\n', ave_PRI, ave_VOI);

fclose(fp);

save('BSDS_test6.mat');
