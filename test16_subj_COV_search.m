function test16_subj_COV_search()
clear
close all

global all_gtlliks all_feats all_gtTree all_gt_sub all_segMap nis iids_sel

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

% specify subj
subj_id = 1109;
iids_sel = iids_ind(ismember(g_subjects, subj_id),:);

%% training 
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

all_gtTree = cell(1,nis);
all_gt_sub = cell(1,nis);
all_segMap = cell(1,nis);

for i = 1:nis
    if iids_sel(:,i) == 0 
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

    all_segMap{i} = segMap;

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
    assert(length(gt_msk) == length(all_gtlliks{i}));
    gt_sub = cell(1,1);
    gt_sub{1}.Segmentation = double(groundTruth{gt_msk}.Segmentation);
    gtTree.llik = all_gtlliks{i}{gt_msk};
        
    all_gtTree{i} = gtTree;
    all_gt_sub{i} = gt_sub;

    % features= [ucm_pa(i) ucm_i sort(ucm(kids(k)) n_leafsUnder(k) dist_to_root(k)]
    if isempty(all_feats{i})
        fdim = 4+1;
        feats_i = zeros(thisTree.numTotalNodes,fdim); 
        %dist_to_r = zeros(thisTree.numTotalNodes,1);
        for k = thisTree.numTotalNodes:-1:thisTree.numLeafNodes+1
            kids = thisTree.getKids(k);
            par = thisTree.getParent(k);
            if par == 0; ucm_par = 1; else ucm_par = thisTree.ucm(par); end
            %dist_to_r(kids) = dist_to_r(k) + 1;
            feats_i(k,:) = [...
                ucm_par,thisTree.ucm(k),sort(thisTree.ucm(kids))',1,...
                %numel(thisTree.leafsUnder{k})/thisTree.numLeafNodes ...
                %dist_to_r(k) ...
            ];
        end
        all_feats{i} = feats_i;
    end
end

% save
if ~exist('all_gtlliks_train.mat', 'file')
    save('all_gtlliks_train.mat', 'all_gtlliks');
end

if ~exist('all_feats_train.mat', 'file')
    save('all_feats_train.mat', 'all_feats');
end

% fminsearch for w in terms of COV
fprintf('\n===== fminsearch init =====\n');
p_g = 0.1;
w0 = [0 0 0 0 log(p_g/(1-p_g))]';
%w0 = glob_search(@eval_cov_rate, w0);
fprintf('===== fminsearch =====\n');
%opts = optimset('Display','iter');
%[wopt, fval] = fminsearch(@eval_cov_rate, w0, opts);
opts = optimset('Display','iter', 'LargeScale', 'off');
[wopt, fval] = fminunc(@eval_cov_rate, w0, opts);
wopt
fval

%%
function COV = eval_cov_rate(w,varargin)
%
global all_feats all_gtTree all_gt_sub all_segMap nis iids_sel

scal = 1e-3;
all_COV = zeros(nis,1);
for i = 1:1%nis
    if iids_sel(:,i) == 0 
        continue 
    end

    %fprintf('%d ', i);
    gtTree = all_gtTree{i};
    feats = all_feats{i};
    gt_sub = all_gt_sub{i};
    segMap = all_segMap{i};

    p = ones(gtTree.numTotalNodes,1);
    tmp = feats*w;
    tmp = sigmf(tmp, [1,0]);

    p(gtTree.numLeafNodes+1:gtTree.numTotalNodes) = tmp(gtTree.numLeafNodes+1:gtTree.numTotalNodes,1);
    p(p == 0) = p(p == 0) + 0.001;
    p(p == 1) = p(p == 1) - 0.001;
    
    [aftTree,segLabels] = inference_temp(gtTree, p, scal);
    
    % eval
    numSegs = length(segLabels);
    labMap = zeros(size(segMap));
    for l = 1:numSegs
        labMap(segMap == l) = segLabels(l);
    end
    el = strel('diamond',1);
    for m = 1:2
       tmp = imdilate(labMap,el);
       labMap(labMap == 0) = tmp(labMap == 0);
    end

    [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
    all_COV(i) = cntR ./ (sumR + (sumR==0));
end

COV = -mean(all_COV(all_COV > 0));
%fprintf('\n');

function w = glob_search(eval_fun, w0)

n_s = 99;
log_ps = linspace(-4.7,-1e-04,n_s); % exp(-2.5) ~ 0.08, exp(-0.1) = 0.9

fvals = zeros(1,n_s);
for k = 1:n_s
    p = exp(log_ps(k));
    w_k = [0 0 0 0 log(p/(1-p))]';

    fvals(k) = eval_fun(w_k);
end

[mCOV, mid] = max(fvals);
p = exp(log_ps(mid));
w = [0 0 0 0 log(p/(1-p))]';
