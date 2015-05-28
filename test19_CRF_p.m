clear
close all

addpath(genpath('external/minFunc_2012'));

dataset = 'train';
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/ucm_trees/' dataset '/'];
pt_dir   = ['./output/ucm_processed_trees/' dataset '/'];
g_subjects = [1102:1117 1119 1121:1124 1126:1130 1132];
iids_train = load(fullfile('data/iids_train.txt')); %

nis = 200;
iids_ind = zeros(length(g_subjects),nis); % 
for i = 1:length(g_subjects)
    s = g_subjects(i);
    load(['data/gt_' num2str(s) '.mat']);
    iids_ind(i,:) = ismember(iids_train, all_iids);
end

%% training 
fdim = 6+1;
subj_id = 1109;
fprintf('===== features =====\n');
if exist('all_gtlliks_train.mat', 'file')
    load('all_gtlliks_train.mat');
else
    all_gtlliks = cell(1,nis); %
end

if exist('all_feats_train.mat', 'file')
    load('all_feats_train.mat');
else
    all_feats = cell(1,nis); %
end

options.Method = 'lbfgs';
options.MaxIter = 100;
options.TolX = 1e-4;
%options.Display = 'full';

all_w = zeros(fdim,nis);
all_fval = zeros(1,nis);
all_eflg = zeros(1,nis);

for i = 1:nis
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

    % gtTree
    if isempty(all_gtlliks{i})
        [gtTree, gt_lliks] = best_gt_trees(segMap, groundTruth, thisTree);
        all_gtlliks{i} = gt_lliks;
    else
        gtTree = thisTree;
    end

    % gt by subj
    subjects = g_subjects(logical(iids_ind(:,i)'));
    gt_msk = ismember(subjects, subj_id); 
    if sum(gt_msk) == 0
        continue
    end
    assert(length(gt_msk) == length(all_gtlliks{i}));
    gt_sub = cell(1,1);
    gt_sub{1}.Segmentation = double(groundTruth{gt_msk}.Segmentation);
    gtTree.llik = all_gtlliks{i}{gt_msk};
        
    % features= [ucm_pa(i) ucm_i sort(ucm(kids(k)) n_leafsUnder(k) dist_to_root(k)]
    if isempty(all_feats{i})
        feats_i = zeros(thisTree.numTotalNodes,fdim); 
        dist_to_r = zeros(thisTree.numTotalNodes,1);
        for k = thisTree.numTotalNodes:-1:thisTree.numLeafNodes+1
            kids = thisTree.getKids(k);
            par = thisTree.getParent(k);
            if par == 0; ucm_par = 1; else ucm_par = thisTree.ucm(par); end
            dist_to_r(kids) = dist_to_r(k) + 1;
            feats_i(k,:) = [...
                ucm_par,thisTree.ucm(k),sort(thisTree.ucm(kids))',...
                numel(thisTree.leafsUnder{k})/thisTree.numLeafNodes, ...
                dist_to_r(k), ...
                1
            ];
        end
        all_feats{i} = feats_i;
    end

    alg_params.scal = 1e-3;
    alg_params.segMap = segMap;
    alg_params.thisTree = gtTree;
    alg_params.gt = gt_sub;
    alg_params.do_eval = 1;

    w0 = randn(fdim,1); w0(end) = rand(1); w0(end) = log(w0(end)/(1-w0(end)));
    [w,fval,exitflag] = minFunc(@func_single, w0, options, alg_params, all_feats{i});
    all_w(:,i) = w;
    all_fval(i) = fval;
    all_eflg(i) = exitflag;
end
