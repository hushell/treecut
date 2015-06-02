clear
close all

dataset = 'train';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

subjects = [1102:1119 1121:1124 1126:1130 1132];
fp = fopen('log_test13.txt', 'w');

for s = subjects 
fprintf('--------------------\n');
fprintf('Subject %d\n', s);
fprintf(fp, '--------------------\n');
fprintf(fp, 'Subject %d\n', s);

generate_subject_data(dataset, s);
load(['data/gt_' num2str(s) '.mat']);

nis = numel(all_iids);
nsegs = numel(all_segs{1});
nsubj = 1; 

scal = 1e-3;
n_r = 1;
n_s = 50;
log_ps = linspace(-2.5,-0.005,n_s); % exp(-2.5) ~ 0.08, exp(-0.1) = 0.9
log_ps = [log_ps log(0.99) log(0.999) log(0.9999) log(0.99999) log(0.999999)];
n_s = n_s + 5;
    
grid_PRI = zeros(n_r,n_s,nsubj,nis);
grid_VOI = zeros(n_r,n_s,nsubj,nis);
grid_nLab = zeros(n_r,n_s,nsubj,nis);
grid_cntR = zeros(n_r,n_s,nsubj,nis);
grid_sumR = zeros(n_r,n_s,nsubj,nis);
grid_COV = zeros(n_r,n_s,nsubj,nis);

cntR_best = 0;
sumR_best = 0;

for i = 1:nis
    % prepare data
    iid = all_iids(i);
    name = num2str(iid);
    load([tree_dir name '_tree.mat']); % tree
    load([ucm2_dir name '.mat']); % ucm2
    %load([gt_dir name '.mat']); % gt
    img = imread([img_dir name '.jpg']); % img
    ucm = ucm2(3:2:end, 3:2:end); % ucm
    segMap = bwlabel(ucm <= 0, 4); % seg
    for j = 1:nsegs
        groundTruth{j}.Segmentation = double(all_segs{i}{j}); % gt
    end
    
    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
    
    % test
    for j = 1:n_s
        p = exp(log_ps(j));
        [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);

        % global
        [aftTree,segLabels] = inference(gtTree, p, scal);
        
        [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
        grid_PRI(1,j,3,i) = PRI;
        grid_VOI(1,j,3,i) = VOI;
        grid_nLab(1,j,3,i) = numel(unique(segLabels));
        
        [cntR, sumR] = covering_rate_ois(labMap, groundTruth);
        COV = cntR ./ (sumR + (sumR==0));
        grid_cntR(1,j,3,i) = cntR;
        grid_sumR(1,j,3,i) = sumR;
        grid_COV(1,j,3,i) = COV;

        %imagesc(labMap);
        %vis_seg(segMap, img, segLabels);
    end % j
        
    fprintf('img %d: best_COV = (%f,%f,%f)\n', i, max(grid_COV(1,:,1,i)));
    fprintf(fp,'img %d: best_COV = (%f,%f,%f)\n', i, max(grid_COV(1,:,1,i)));
end % i 

[COV_g, p_g] = max(sum(grid_COV(1,:,3,:),4));
p_g = exp(log_ps(p_g));

fprintf('COV_g = %f, p_g = %f; \n', COV_g, p_g);
fprintf(fp, 'COV_g = %f, p_g = %f; \n', COV_g, p_g);

all_aftTree = cell(nis,nsubj);
for i = 1:nis
    % prepare data
    iid = all_iids(i);
    name = num2str(iid);
    load([tree_dir name '_tree.mat']); % tree
    load([ucm2_dir name '.mat']); % ucm2
    %load([gt_dir name '.mat']); % gt
    img = imread([img_dir name '.jpg']); % img
    ucm = ucm2(3:2:end, 3:2:end); % ucm
    segMap = bwlabel(ucm <= 0, 4); % seg
    for j = 1:nsegs
        groundTruth{j}.Segmentation = double(all_segs{i}{j}); % gt
    end
    
    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
    
    [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);

    % global
    [aftTree,segLabels] = inference(gtTree, p_g, scal);
    all_aftTree{i,3} = aftTree;
end

save(['BSDS_test13_' num2str(s) '.mat']);
end

fclose(fp);

