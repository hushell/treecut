clear
close all
iids_train = load(fullfile('data/iids_train.txt')); %

fp = fopen('log_test21.txt', 'w');

%% train p for TC and k for UCM
dataset = 'train';
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

n_sub = 2;
nis = 50;

% TC
%scals = [1e-2 1e-3 5e-4 1e-4 1e-5];
%scals = [9e-4 8e-4 7e-4 6e-4];
scals = [2e-4 3e-4 4e-4];
n_r = length(scals);
%ps = linspace(0.9,0.9999,10);
ps = [exp(linspace(log(0.80), log(0.89), 30)) exp(linspace(log(0.9), log(0.9999), 30))];
n_s = length(ps);

grid_PRI = zeros(n_r,n_s,n_sub,nis,2);
grid_VOI = zeros(n_r,n_s,n_sub,nis,2);
grid_nLab = zeros(n_r,n_s,n_sub,nis,2);
grid_cntR = zeros(n_r,n_s,n_sub,nis,2);
grid_sumR = zeros(n_r,n_s,n_sub,nis,2);
grid_COV = zeros(n_r,n_s,n_sub,nis,2);

if exist('all_gtlliks_train.mat', 'file')
    load('all_gtlliks_train.mat');
else
    all_gtlliks = cell(1,nis); %
end

el = strel('diamond',1);

for i = 1:nis
    tic;
    ii = i+0; %*****
    iid = iids_train(ii); 
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

    if isempty(all_gtlliks{ii})
        [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);
        all_gtlliks{ii} = gt_lliks;
    else
        gtTree = thisTree;
    end

    for s = 1:2
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{ind_min_max(s)}.Segmentation);

%         % TC-GT
%         gtTree.llik = all_gtlliks{ii}{ind_min_max(s)};
%         
%         for j = 1:n_s
%             for r = 1:n_r
%                 p = ps(j);
%                 scal = scals(r);
% 
%                 [aftTree,segLabels] = inference_temp(gtTree, p, scal); % gtTree
%                 
%                 [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
%                 grid_PRI(r,j,s,i,1) = PRI;
%                 grid_VOI(r,j,s,i,1) = VOI;
%                 grid_nLab(r,j,s,i,1) = numel(unique(segLabels));
%                 
%                 [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
%                 COV = cntR ./ (sumR + (sumR==0));
%                 grid_cntR(r,j,s,i,1) = cntR;
%                 grid_sumR(r,j,s,i,1) = sumR;
%                 grid_COV(r,j,s,i,1) = COV;
%             end
%         end % j
% 
%         fprintf('TC-GT: (img %d, sub %d): best_COV = %f\n', ii, s, max(max(grid_COV(:,:,s,i,1))));
%         fprintf(fp, 'TC-GT: (img %d, sub %d): best_COV = %f\n', ii, s, max(max(grid_COV(:,:,s,i,1))));
            
        % TC
        for j = 1:n_s
            for r = 1:n_r
                p = ps(j);
                scal = scals(r);

                [aftTree,segLabels] = inference_temp(thisTree, p, scal); % thisTree
                
                [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
                grid_PRI(r,j,s,i,2) = PRI;
                grid_VOI(r,j,s,i,2) = VOI;
                grid_nLab(r,j,s,i,2) = numel(unique(segLabels));
                
                [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
                COV = cntR ./ (sumR + (sumR==0));
                grid_cntR(r,j,s,i,2) = cntR;
                grid_sumR(r,j,s,i,2) = sumR;
                grid_COV(r,j,s,i,2) = COV;
            end
        end % j

        fprintf('TC: (img %d, sub %d): best_COV = %f\n', ii, s, max(max(grid_COV(:,:,s,i,2))));
        fprintf(fp, 'TC: (img %d, sub %d): best_COV = %f\n', ii, s, max(max(grid_COV(:,:,s,i,2))));
    end % s

    fprintf('img %d takes %f sec.\n', iid, toc);
end % i

% select ODS
COV_subs = zeros(2,n_sub);
p_subs = zeros(2,n_sub);
best_j = zeros(2,n_sub);
scal_subs = zeros(2,n_sub);
best_r = zeros(2,n_sub);
for s = 1:n_sub
    sub_COV = squeeze(sum(grid_COV(:,:,s,:,1),4));
    [COV_g, I_g] = max(sub_COV(:));
    [r_g,j_g] = ind2sub(size(sub_COV),I_g);
    scal_g = scals(r_g);
    p_g = ps(j_g);
    COV_subs(1,s) = COV_g / nis;
    best_j(1,s) = j_g;
    p_subs(1,s) = p_g;
    best_r(1,s) = r_g;
    scal_subs(1,s) = scal_g;
    fprintf('TC-GT, sub %d: COV_g = %f, p_g = %f, scal_g = %f; \n', s, COV_subs(1,s), p_g, scal_g);

    sub_COV = squeeze(sum(grid_COV(:,:,s,:,2),4));
    [COV_g, I_g] = max(sub_COV(:));
    [r_g,j_g] = ind2sub(size(sub_COV),I_g);
    scal_g = scals(r_g);
    p_g = ps(j_g);
    COV_subs(2,s) = COV_g / nis;
    best_j(2,s) = j_g;
    p_subs(2,s) = p_g;
    best_r(2,s) = r_g;
    scal_subs(1,s) = scal_g;
    fprintf('TC, sub %d: COV_g = %f, p_g = %f, scal_g = %f; \n', s, COV_subs(2,s), p_g, scal_g);
end

% test
test_COV = zeros(2,n_sub,n_sub,nis);
for i = 1:nis
    for s = 1:n_sub % use GT s
        for r = 1:n_sub % use param r
            test_COV(1,r,s,i) = grid_COV(best_r(1,r),best_j(1,r),s,i,1); % TC-GT
            test_COV(2,r,s,i) = grid_COV(best_r(2,r),best_j(2,r),s,i,2); % TC
        end % r
    end % s
end % i

squeeze(mean(test_COV(1,:,:,:),4))
squeeze(mean(test_COV(2,:,:,:),4))

save BSDS_test21.mat

