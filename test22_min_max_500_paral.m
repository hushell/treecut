function test22_min_max_500_paral()
clear
close all

run_test22('train');
run_test22('test');

function run_test22(dataset)
if nargin < 1
    dataset = 'train';
end

n_sub = 2;
n_alg = 2;

% TC
scals = [1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4];
n_r = length(scals);
ps = [exp(linspace(log(0.0001), log(0.09), 5)) exp(linspace(log(0.1), log(0.79), 35)) exp(linspace(log(0.8), log(0.89), 30)) exp(linspace(log(0.9), log(0.9999), 30))];
n_s = length(ps);

% UCM
g_thres = 0.01:0.01:1.00;
n_th = length(g_thres);

%% 
fprintf('===== %s set =====\n', dataset);
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

nis = length(all_files);

grid_PRI = zeros(n_r,n_s,n_sub,nis,n_alg);
grid_VOI = zeros(n_r,n_s,n_sub,nis,n_alg);
grid_nLab = zeros(n_r,n_s,n_sub,nis,n_alg);
grid_cntR = zeros(n_r,n_s,n_sub,nis,n_alg);
grid_sumR = zeros(n_r,n_s,n_sub,nis,n_alg);
grid_COV = zeros(n_r,n_s,n_sub,nis,n_alg);

el = strel('diamond',1);

for i = 1:nis
    tic;
    ii = i+0; %*****
    [~,name] = fileparts(all_files(ii).name);
    iid = str2num(name);
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

    for s = 1:2
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{ind_min_max(s)}.Segmentation);

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
            end % r
        end % j

        fprintf('TC: (img %d, sub %d): best_COV = %f\n', ii, s, max(max(grid_COV(:,:,s,i,2))));

        % UCM
        for j = 1:n_s
            k = g_thres(j);

            labMap = bwlabel(ucm <= k, 4);
            
            for m = 1:2
               tmp = imdilate(labMap,el);
               labMap(labMap == 0) = tmp(labMap == 0);
            end
            
            [PRI, VOI] = match_segmentations2(labMap, gt_sub);
            grid_PRI(1,j,s,i,1) = PRI;
            grid_VOI(1,j,s,i,1) = VOI;
            grid_nLab(1,j,s,i,1) = numel(unique(labMap(:)));
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            grid_cntR(1,j,s,i,1) = cntR;
            grid_sumR(1,j,s,i,1) = sumR;
            grid_COV(1,j,s,i,1) = COV;
        end % j

        fprintf('UCM: (img %d, sub %d): best_COV = %f\n', ii, s, max(max(grid_COV(:,:,s,i,1))));
    end % s
end % i

save(['BSDS_test22_' dataset '.mat']); % **** IMPORTANT

