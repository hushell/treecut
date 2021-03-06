clear
close all

dataset = 'train';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

load('data/gt_1105_1123.mat');

nis = numel(all_iids);
nsegs = numel(all_segs{1});
nsubj = 2+1; % global, subj1, subj2

fp = fopen('log_test12.txt', 'w');

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
    for s = 1:nsegs
        groundTruth{s}.Segmentation = double(all_segs{i}{s}); % gt
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

        % subj1
        gtTree.llik = gt_lliks{1};
        [aftTree,segLabels] = inference(gtTree, p, scal);
        
        [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth(1));
        grid_PRI(1,j,1,i) = PRI;
        grid_VOI(1,j,1,i) = VOI;
        grid_nLab(1,j,1,i) = numel(unique(segLabels));
        
        [cntR, sumR] = covering_rate_ois(labMap, groundTruth(1));
        COV = cntR ./ (sumR + (sumR==0));
        grid_cntR(1,j,1,i) = cntR;
        grid_sumR(1,j,1,i) = sumR;
        grid_COV(1,j,1,i) = COV;

        % subj2
        gtTree.llik = gt_lliks{2};
        [aftTree,segLabels] = inference(gtTree, p, scal);
        
        [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth(2));
        grid_PRI(1,j,2,i) = PRI;
        grid_VOI(1,j,2,i) = VOI;
        grid_nLab(1,j,2,i) = numel(unique(segLabels));
        
        [cntR, sumR] = covering_rate_ois(labMap, groundTruth(2));
        COV = cntR ./ (sumR + (sumR==0));
        grid_cntR(1,j,2,i) = cntR;
        grid_sumR(1,j,2,i) = sumR;
        grid_COV(1,j,2,i) = COV;

        %imagesc(labMap);
        %vis_seg(segMap, img, segLabels);
    end % j
        
    fprintf('img %d: best_COV = (%f,%f,%f)\n', i, max(grid_COV(1,:,1,i)),max(grid_COV(1,:,2,i)),max(grid_COV(1,:,3,i)));
    fprintf(fp, 'img %d: best_COV = (%f,%f,%f)\n', i, max(grid_COV(1,:,1,i)),max(grid_COV(1,:,2,i)),max(grid_COV(1,:,3,i)));
end % i 

grid_COV(1,:,4,:) = grid_COV(1,:,1,:)/2 + grid_COV(1,:,2,:)/2;

[COV_g, I_p_g] = max(sum(grid_COV(1,:,3,:),4));
[COV_1, I_p_1] = max(sum(grid_COV(1,:,1,:),4));
[COV_2, I_p_2] = max(sum(grid_COV(1,:,2,:),4));
[COV_s, I_p_s] = max(sum(grid_COV(1,:,4,:),4));

p_g = exp(log_ps(I_p_g));
p_1 = exp(log_ps(I_p_1));
p_2 = exp(log_ps(I_p_2));

fprintf('COV_g = %f, p_g = %f; COV_1 = %f, p_1 = %f; COV_2 = %f, p_2 = %f;\n', COV_g, p_g, COV_1, p_1, COV_2, p_2);
fprintf(fp, 'COV_g = %f, p_g = %f; COV_1 = %f, p_1 = %f; COV_2 = %f, p_2 = %f;\n', COV_g, p_g, COV_1, p_1, COV_2, p_2);

all_covs = zeros(nis,4);
all_covs(:,1) = grid_COV(1,I_p_1,1,:);
all_covs(:,2) = grid_COV(1,I_p_2,2,:);
all_covs(:,3) = grid_COV(1,I_p_g,3,:);
all_covs(:,4) = grid_COV(1,I_p_s,4,:);
figure; plot(1:38, all_covs(:,1), 'o-', 1:38, all_covs(:,2), '+-', 1:38, all_covs(:,3), '*-')
figure; plot(1:38, all_covs(:,1), 'o-', 1:38, all_covs(:,2), '+-', 1:38, all_covs(:,4), '^-')


all_aftTree = cell(nis,3);
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
    for s = 1:nsegs
        groundTruth{s}.Segmentation = double(all_segs{i}{s}); % gt
    end
    
    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);
    
    [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);

    % global
    [aftTree,segLabels] = inference(gtTree, p_g, scal);
    all_aftTree{i,3} = aftTree;
    
    % subj1
    gtTree.llik = gt_lliks{1};
    [aftTree,segLabels] = inference(gtTree, p_1, scal);
    all_aftTree{i,1} = aftTree;
    
    % subj2
    gtTree.llik = gt_lliks{1};
    [aftTree,segLabels] = inference(gtTree, p_2, scal);
    all_aftTree{i,2} = aftTree;
end

fclose(fp);

save('BSDS_test12.mat');

all_lliks = zeros(38,3);
all_maps = zeros(38,3);
for i = 1:3
for j = 1:38
all_lliks(j,i) = all_aftTree{j,i}.E(end);
all_maps(j,i) = all_aftTree{j,i}.M(end);
end
end
figure; plot(1:38, all_lliks(:,1), 'o-', 1:38, all_lliks(:,2), '+-', 1:38, all_lliks(:,3), '*-')
figure; plot(1:38, all_maps(:,1), 'o-', 1:38, all_maps(:,2), '+-', 1:38, all_maps(:,3), '*-')
