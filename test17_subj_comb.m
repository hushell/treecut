clear
close all

g_subjects = [1102:1117 1119 1121:1124 1126:1130 1132];
iids_train = load(fullfile('data/iids_train.txt')); %

nis = 200;
iids_ind = zeros(length(g_subjects),nis); % 
for i = 1:length(g_subjects)
    s = g_subjects(i);
    load(['data/gt_' num2str(s) '.mat']);
    iids_ind(i,:) = ismember(iids_train, all_iids);
end
% imagesc(iids_ind); colormap gray
% figure; plot(1:nis, sum(iids_ind,1), '+');

% subject combinations
n_sub = 3;
counts = sum(iids_ind,1);
combs = combnk(g_subjects,n_sub);
%combs = iids_ind(:, counts == n_sub);
%combs = unique(combs', 'rows')';

cnt_combs = zeros(size(combs,1),1);

for k = 1:size(combs,1)
    subs = ismember(g_subjects, combs(k,:))';
    iids_sel = iids_ind & repmat(subs, [1,nis]);
    iids_sel = sum(iids_sel,1);
    iids_sel = iids_sel(iids_sel >= n_sub);
    %fprintf('# imgs = %d\n', sum(iids_sel));
    cnt_combs(k) = numel(iids_sel);
end

candidates = combs(cnt_combs>20,:); 
sub_sel = candidates(3,:); % 1105 1109 1123 %
subs = ismember(g_subjects, sub_sel)';
iids_sel = iids_ind(subs,:); % 
iids_inter = sum(iids_sel,1) == n_sub; %

%% train p for TC and k for UCM
dataset = 'train';
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

% TC
scal = 1e-3;
n_r = 2; % here as num of methods
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
    if sum(iids_sel(:,i)) == 0 || sum(iids_sel(:,i)) == n_sub 
        continue % not in intersection
    end

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
    
    subjects = g_subjects(logical(iids_ind(:,i)'));

    if isempty(all_gtlliks{i})
        [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);
        all_gtlliks{i} = gt_lliks;
    else
        gtTree = thisTree;
    end

    for s = 1:n_sub
        gt_msk = ismember(subjects, sub_sel(s)); 
        if sum(gt_msk) == 0 % i doesn't have subj s
            continue
        end
        assert(length(gt_msk) == length(all_gtlliks{i}));
        
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{gt_msk}.Segmentation);

        gtTree.llik = all_gtlliks{i}{gt_msk};
        
        % TC
        for j = 1:n_s
            p = exp(log_ps(j));

            % global
            [aftTree,segLabels] = inference_temp(gtTree, p, scal);
            
            [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
            grid_PRI(1,j,s,i) = PRI;
            grid_VOI(1,j,s,i) = VOI;
            %grid_nLab(1,j,s,i) = numel(unique(segLabels));
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            grid_cntR(1,j,s,i) = cntR;
            grid_sumR(1,j,s,i) = sumR;
            grid_COV(1,j,s,i) = COV;
        end % j

        fprintf('img %d, sub %d: best_COV_TC = %f\n', i, sub_sel(s), max(grid_COV(1,:,s,i)));
            
        % UCM
        for j = 1:n_th
            k = g_thres(j);
            labMap = bwlabel(ucm <= k, 4);
            
            for m = 1:2
               tmp = imdilate(labMap,el);
               labMap(labMap == 0) = tmp(labMap == 0);
            end
            
            [PRI, VOI] = match_segmentations2(labMap, gt_sub);
            grid_PRI(2,j,s,i) = PRI;
            grid_VOI(2,j,s,i) = VOI;
            %grid_nLab(2,j,s,i) = numel(unique(segLabels));
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            grid_cntR(2,j,s,i) = cntR;
            grid_sumR(2,j,s,i) = sumR;
            grid_COV(2,j,s,i) = COV;
        end % j

        fprintf('img %d, sub %d: best_COV_UCM = %f\n', i, sub_sel(s), max(grid_COV(2,:,s,i)));
    end % s
end % i

% select ODS
COV_subs = zeros(2,n_sub);
p_subs = zeros(2,n_sub);
best_j = zeros(2,n_sub);
for s = 1:n_sub
    [COV_g, j_g] = max(sum(grid_COV(1,:,s,:),4));
    p_g = exp(log_ps(j_g));
    COV_subs(1,s) = COV_g;
    best_j(1,s) = j_g;
    p_subs(1,s) = p_g;
    fprintf('TC, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);

    [COV_g, j_g] = max(sum(grid_COV(2,:,s,:),4));
    p_g = g_thres(j_g);
    COV_subs(2,s) = COV_g;
    best_j(2,s) = j_g;
    p_subs(2,s) = p_g;
    fprintf('UCM, sub %d: COV_g = %f, p_g = %f; \n', sub_sel(s), COV_g, p_g);
end

% test
test_COV = zeros(n_r,n_sub,n_sub,nis);
for i = 1:nis
    if sum(iids_sel(:,i)) ~= n_sub 
        continue
    end

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
    
    subjects = g_subjects(logical(iids_ind(:,i)'));

    if isempty(all_gtlliks{i})
        [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);
        all_gtlliks{i} = gt_lliks;
    else
        gtTree = thisTree;
    end

    for s = 1:n_sub
        gt_msk = ismember(subjects, sub_sel(s));
        assert(sum(gt_msk) == 1);
        assert(length(gt_msk) == length(all_gtlliks{i}));
        
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{gt_msk}.Segmentation);

        gtTree.llik = all_gtlliks{i}{gt_msk};

        for r = 1:n_sub
            % TC
            [aftTree,segLabels] = inference_temp(gtTree, p_subs(1,r), scal);
            [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            test_COV(1,r,s,i) = COV;

            % UCM
            labMap = bwlabel(ucm <= p_subs(2,r), 4);
            for m = 1:2
               tmp = imdilate(labMap,el);
               labMap(labMap == 0) = tmp(labMap == 0);
            end
            [PRI, VOI] = match_segmentations2(labMap, gt_sub);
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            test_COV(2,r,s,i) = COV;
        end % r
    end % s
end % i

test_COV = test_COV(:,:,:,iids_inter);
squeeze(mean(test_COV(1,:,:,:),4))
squeeze(mean(test_COV(2,:,:,:),4))

% paired t-test
H = zeros(n_sub);
p_value = zeros(n_sub);
for s = 1:n_sub
    for r = 1:n_sub
        [H(r,s), p_value(r,s)] = ttest2(test_COV(1,r,s,:), test_COV(2,r,s,:));
    end
end
H
p_value

% save
if ~exist('all_gtlliks_train.mat', 'file')
    save('all_gtlliks_train.mat', 'all_gtlliks');
end

