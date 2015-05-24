clear
close all

fp = fopen('log_test20.txt', 'w');

%% train p for TC and k for UCM
dataset = 'train';
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

n_r = 3; % TC-GT TC UCM 
n_sub = 2;

% TC
scal = 1e-3;
n_s = 99;
log_ps = linspace(-4.7,-1e-04,n_s); % exp(-2.5) ~ 0.08, exp(-0.1) = 0.9

% UCM
n_th = 99;
g_thres = 0.01:0.01:0.99;

grid_PRI = zeros(n_r,max(n_s,n_th),n_sub,nis);
grid_VOI = zeros(n_r,max(n_s,n_th),n_sub,nis);
grid_nLab = zeros(n_r,max(n_s,n_th),n_sub,nis);
grid_cntR = zeros(n_r,max(n_s,n_th),n_sub,nis);
grid_sumR = zeros(n_r,max(n_s,n_th),n_sub,nis);
grid_COV = zeros(n_r,max(n_s,n_th),n_sub,nis);

if exist('all_gtlliks_train.mat', 'file')
    load('all_gtlliks_train.mat');
else
    all_gtlliks = cell(1,nis); %
end

el = strel('diamond',1);

for i = 1:nis
    tic;
    iid = iids_train(i);
    name = num2str(iid);
    load([tree_dir name '_tree.mat']); % tree thres_arr
    load([ucm2_dir name '.mat']); % ucm2
    load([gt_dir name '.mat']); % gt
    img = imread([img_dir name '.jpg']); % img
    ucm = ucm2(3:2:end, 3:2:end); % ucm
    segMap = bwlabel(ucm <= 0, 4); % seg

    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);

    % select min & max GT
    gt_lab_cnt = cellfun(@(x) numel(unique(x.Segmentation(:))), groundTruth);
    [~,ind_lab_cnt] = sort(gt_lab_cnt);
    ind_min_max = [ind_lab_cnt(1) ind_lab_cnt(end)];

    if isempty(all_gtlliks{i})
        [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);
        all_gtlliks{i} = gt_lliks;
    else
        gtTree = thisTree;
    end

    for s = 1:2
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{ind_min_max(s)}.Segmentation);

        % TC-GT
        gtTree.llik = all_gtlliks{i}{ind_min_max(s)};
        
        for j = 1:n_s
            p = exp(log_ps(j));

            [aftTree,segLabels] = inference_temp(gtTree, p, scal); % gtTree
            
            [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
            grid_PRI(1,j,s,i) = PRI;
            grid_VOI(1,j,s,i) = VOI;
            grid_nLab(1,j,s,i) = numel(unique(segLabels));
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            grid_cntR(1,j,s,i) = cntR;
            grid_sumR(1,j,s,i) = sumR;
            grid_COV(1,j,s,i) = COV;
        end % j

        fprintf('TC-GT: (img %d, sub %d): best_COV = %f\n', i, s, max(grid_COV(1,:,s,i)));
        fprintf(fp, 'TC-GT: (img %d, sub %d): best_COV = %f\n', i, s, max(grid_COV(1,:,s,i)));
            
        % TC
        for j = 1:n_s
            p = exp(log_ps(j));

            [aftTree,segLabels] = inference_temp(thisTree, p, scal); % thisTree
            
            [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
            grid_PRI(2,j,s,i) = PRI;
            grid_VOI(2,j,s,i) = VOI;
            grid_nLab(2,j,s,i) = numel(unique(segLabels));
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            grid_cntR(2,j,s,i) = cntR;
            grid_sumR(2,j,s,i) = sumR;
            grid_COV(2,j,s,i) = COV;
        end % j

        fprintf('TC: (img %d, sub %d): best_COV = %f\n', i, s, max(grid_COV(2,:,s,i)));
        fprintf(fp, 'TC: (img %d, sub %d): best_COV = %f\n', i, s, max(grid_COV(2,:,s,i)));

        % UCM
        for j = 1:n_th
            k = g_thres(j);
            labMap = bwlabel(ucm <= k, 4);
            
            for m = 1:2
               tmp = imdilate(labMap,el);
               labMap(labMap == 0) = tmp(labMap == 0);
            end
            
            [PRI, VOI] = match_segmentations2(labMap, gt_sub);
            grid_PRI(3,j,s,i) = PRI;
            grid_VOI(3,j,s,i) = VOI;
            grid_nLab(3,j,s,i) = numel(unique(labMap(:)));
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            grid_cntR(3,j,s,i) = cntR;
            grid_sumR(3,j,s,i) = sumR;
            grid_COV(3,j,s,i) = COV;
        end % j

        fprintf('UCM: (img %d, sub %d): best_COV = %f\n', i, s, max(grid_COV(3,:,s,i)));
        fprintf(fp, 'UCM: (img %d, sub %d): best_COV = %f\n', i, s, max(grid_COV(3,:,s,i)));
    end % s

    fprintf('img %d takes %f sec.\n', i, toc);
end % i

% select ODS
COV_subs = zeros(n_r,n_sub);
p_subs = zeros(n_r,n_sub);
best_j = zeros(n_r,n_sub);
for s = 1:n_sub
    [COV_g, j_g] = max(sum(grid_COV(1,:,s,:),4));
    p_g = exp(log_ps(j_g));
    COV_subs(1,s) = COV_g;
    best_j(1,s) = j_g;
    p_subs(1,s) = p_g;
    fprintf('TC-GT, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);
    fprintf(fp, 'TC-GT, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);

    [COV_g, j_g] = max(sum(grid_COV(2,:,s,:),4));
    p_g = g_thres(j_g);
    COV_subs(2,s) = COV_g;
    best_j(2,s) = j_g;
    p_subs(2,s) = p_g;
    fprintf('TC, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);
    fprintf(fp, 'TC, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);

    [COV_g, j_g] = max(sum(grid_COV(3,:,s,:),4));
    p_g = g_thres(j_g);
    COV_subs(3,s) = COV_g;
    best_j(3,s) = j_g;
    p_subs(3,s) = p_g;
    fprintf('UCM, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);
    fprintf(fp, 'UCM, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);
end

% test
test_COV = zeros(n_r,n_sub,n_sub,nis);
for i = 1:nis
    for s = 1:n_sub % use GT s
        for r = 1:n_sub % use param r
            test_COV(1,r,s,i) = grid_COV(1,best_j(1,r),s,i); % TC-GT
            test_COV(2,r,s,i) = grid_COV(2,best_j(2,r),s,i); % TC
            test_COV(3,r,s,i) = grid_COV(3,best_j(3,r),s,i); % UCM
        end % r
    end % s
end % i

squeeze(mean(test_COV(1,:,:,:),4))
squeeze(mean(test_COV(2,:,:,:),4))

% paired t-test
H = zeros(n_sub);
p_value = zeros(n_sub);
for s = 1:n_sub
    for r = 1:n_sub
        [H(r,s), p_value(r,s)] = ttest2(test_COV(2,r,s,:), test_COV(3,r,s,:));
    end
end
H
p_value

% save
if ~exist('all_gtlliks_train.mat', 'file')
    save('all_gtlliks_train.mat', 'all_gtlliks');
end

save BSDS_test20.mat
fclose(fp);

