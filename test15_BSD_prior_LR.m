clear
close all
addpath external/netlab3_3/

% load previous results
load BSDS_test6.mat
load BSDS_test15.mat

dataset = 'test';

img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/ucm_trees/' dataset '/'];
pt_dir   = ['./output/ucm_processed_trees/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

for iter = 1:5
fprintf('================ iteration %d ===============\n', iter);
fdim = 5;
X = zeros(300000,fdim);
Y = zeros(300000,1);
TI = zeros(300000,1);
pt  = 1;

%all_gtlliks = cell(1,numel(all_files));
%all_gtallik = cell(1,numel(all_files));
for i = 1:200%numel(all_files)
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
    
    % gtTree
    %p = exp(log_ps(all_best_j_COV(i)));
    %[gtTree, gt_lliks, gt_labs] = best_gt_trees(segMap, groundTruth, thisTree);
    %gtTree.llik = gt_lliks{1};
    %all_gtallik{i} = gtTree.llik;
    %all_gtlliks{i} = gt_lliks;
    gtTree = thisTree;
    gtTree.llik = all_gtallik{i};
    
    dist_to_r = zeros(thisTree.numTotalNodes,1);
    ucm_k_kids = zeros(thisTree.numTotalNodes,fdim); 
    for k = thisTree.numTotalNodes:-1:thisTree.numLeafNodes+1
        kids = thisTree.getKids(k);
        dist_to_r(kids) = dist_to_r(k) + 1;
        % features
        ucm_k_kids(k,:) = [
            thisTree.ucm(k), thisTree.ucm(kids)' ...
            numel(thisTree.leafsUnder{k})/thisTree.numLeafNodes ...
            dist_to_r(k) ...
            ];
    end
    
    if iter == 1
        %p = 0.7;
        p = exp(log_ps(all_best_j_COV(i)));
    else
        p = ones(thisTree.numTotalNodes,1);
        tmp = glmfwd(net, ucm_k_kids);
        p(gtTree.numLeafNodes+1:gtTree.numTotalNodes) = tmp(gtTree.numLeafNodes+1:gtTree.numTotalNodes,1);
        p(p == 0) = p(p == 0) + 0.001;
        p(p == 1) = p(p == 1) - 0.001;
    end
    [aftTree,segLabels] = inference_temp(gtTree, p, scal);
    
    %[PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
    %[cntR, sumR] = covering_rate_ois(labMap, groundTruth);
    %COV = cntR ./ (sumR + (sumR==0));
    %fprintf('img %d: COV_g = %f, COV_now = %f\n', i, all_grid_R{i}(all_best_j_COV(i)), COV);
    
    % A+\A: 1  A-: 0  Leafs: 3  A: pos_ind
    govern = zeros(aftTree.numTotalNodes,1);
    for k = aftTree.numTotalNodes:-1:aftTree.numLeafNodes+1
        kids = aftTree.getKids(k);
        if aftTree.activeNodes(k) == 1 || govern(k) == 1
            govern(kids) = 1; % all nodes below active nodes are govern, indicated as 1
        end
    end
    govern(1:aftTree.numLeafNodes) = 3;
    
%     pos_ind = 1; % model 1: A+ vs. A-
    pos_ind = 2; % model 2: A vs. A-
    govern(~govern & aftTree.activeNodes) = pos_ind;

    feat_pos = ucm_k_kids(govern == pos_ind, :);
    y_tree = ones(size(feat_pos,1),1);
    
    feat_neg = ucm_k_kids(govern == 0, :);
    if i <= 100 && iter == 1
        %len = min(sum(govern==0), sum(govern==pos_ind)); 
        %len = floor(sum(govern==0)*0.5);
        %feat_neg = feat_neg(1:len,:);
    end
    y_tree = [y_tree; 2*ones(size(feat_neg,1),1)];
    
    X(pt:pt+length(y_tree)-1,:) = [feat_pos; feat_neg];
    Y(pt:pt+length(y_tree)-1) = y_tree;

    if i <= 100
        TI(pt:pt+length(y_tree)-1) = 1;
    end
    
    pt = pt+length(y_tree);

    % model 3: CRF-like
end
fprintf('\n');

X = X(1:pt-1,:);
Y = Y(1:pt-1,:);
TI = logical(TI(1:pt-1,:));

Xte = X(~TI,:);
Yte = Y(~TI,:);
Xtr = X(TI,:);
Ytr = Y(TI,:);

pPos = sum(Ytr == 1) / length(Ytr);
pNeg = sum(Ytr == 2) / length(Ytr);
Alpha = Ytr;
Alpha(Ytr == 1) = (1/2)/pPos;
Alpha(Ytr == 2) = (1/2)/pNeg;

net = train_lr(Xtr, Ytr, Alpha);
[cl, Z, pcorr, acc] = test_lr(net, Xte, Yte);

%%
all_metrics = zeros(3,numel(all_files)); % PRI, VOI, COV
for i = 101:105
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
    aftTree = thisTree;
    
    % gtTree
    gtTree = thisTree;
    gtTree.llik = all_gtallik{i};
    
    p = ones(aftTree.numTotalNodes,1);
    dist_to_r = zeros(aftTree.numTotalNodes,1);
    ucm_k_kids = zeros(aftTree.numTotalNodes,fdim); 
    for k = aftTree.numTotalNodes:-1:aftTree.numLeafNodes+1
        kids = aftTree.getKids(k);
        dist_to_r(kids) = dist_to_r(k) + 1;
        % features
        ucm_k_kids(k,:) = [
            aftTree.ucm(k), aftTree.ucm(kids)' ...
            numel(aftTree.leafsUnder{k})/aftTree.numLeafNodes ...
            dist_to_r(k) ...
            ];
    end
    tmp = glmfwd(net, ucm_k_kids);
    p(aftTree.numLeafNodes+1:aftTree.numTotalNodes) = tmp(aftTree.numLeafNodes+1:aftTree.numTotalNodes,1);
    p(p == 0) = p(p == 0) + 0.001;
    p(p == 1) = p(p == 1) - 0.001;
    
    [aftTree,segLabels] = inference_temp(gtTree, p, scal);
    
    [PRI, VOI, labMap] = eval_seg(segMap, segLabels, groundTruth);
    [cntR, sumR] = covering_rate_ois(labMap, groundTruth);
    COV = cntR ./ (sumR + (sumR==0));
    all_metrics(:,i) = [PRI; VOI; COV];
    
    fprintf('img %d: COV_g = %f, COV_now = %f\n', i, all_grid_R{i}(all_best_j_COV(i)), COV);
end
fprintf('\n');

end % iter