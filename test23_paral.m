function test23_paral(dataset,img_s, img_t)
if nargin < 3
    dataset = 'train';
    img_range = [1 2];
end

img_range = [str2num(img_s) str2num(img_t)];

fprintf('===== %s set =====\n', dataset);
fprintf('===== %d to %d=====\n', img_range(1), img_range(end));

%% 
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
eval_dir  = ['./output/grid_eval/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

nis = length(all_files);
el = strel('diamond',1);

%if matlabpool('size') == 0 % checking to see if my pool is already open
%    matlabpool open 8
%end

for i = img_range
    fprintf('===== img %d =====\n', i);
    n_alg = 2;

    % TC
    scals = [1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4];
    n_r = length(scals);
    ps = [exp(linspace(log(0.0001), log(0.09), 5)) exp(linspace(log(0.1), log(0.79), 35)) exp(linspace(log(0.8), log(0.89), 30)) exp(linspace(log(0.9), log(0.9999), 30))];
    n_s = length(ps);

    % UCM
    thres = 0.01:0.01:1.00;
    
    % data
    [~,name] = fileparts(all_files(i).name);
    %iid = str2double(name);
    temp = load([tree_dir name '_tree.mat']); % tree thres_arr
    thisTree = temp.thisTree;
    temp = load([ucm2_dir name '.mat']); % ucm2
    ucm2 = temp.ucm2;
    temp = load([gt_dir name '.mat']); % gt
    groundTruth = temp.groundTruth;
    img = imread([img_dir name '.jpg']); % img
    ucm = ucm2(3:2:end, 3:2:end); % ucm
    segMap = bwlabel(ucm <= 0, 4); % seg

    % preprocess tree
    thisTreePath = [pt_dir name '_tree.mat'];
    thisTree = tree_preprocess(thisTreePath, thisTree, img, segMap);

    %% select min & max GT
    %gt_lab_cnt = cellfun(@(x) numel(unique(x.Segmentation(:))), groundTruth);
    %[~,ind_lab_cnt] = sort(gt_lab_cnt);
    %ind_min_max = [ind_lab_cnt(1) ind_lab_cnt(end)];
    n_sub = length(groundTruth);
    
    grid_PRI = zeros(n_r,n_s,n_sub,n_alg);
    grid_VOI = zeros(n_r,n_s,n_sub,n_alg);
    grid_nLab = zeros(n_r,n_s,n_sub,n_alg);
    grid_COV = zeros(n_r,n_s,n_sub,n_alg);

    for s = 1:n_sub
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{s}.Segmentation);

        % TC
        for j = 1:n_s
            for r = 1:n_r
                p = ps(j);
                scal = scals(r);

                [~,segLabels] = inference_temp(thisTree, p, scal); % thisTree
                
                [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
                grid_PRI(r,j,s,2) = PRI;
                grid_VOI(r,j,s,2) = VOI;
                grid_nLab(r,j,s,2) = numel(unique(segLabels));
                
                [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
                COV = cntR ./ (sumR + (sumR==0));
                grid_COV(r,j,s,2) = COV;
            end % r
        end % j

        fprintf('TC: (img %d, sub %d): best_COV = %f\n', i, s, max(max(grid_COV(:,:,s,2))));

        % UCM
        for j = 1:n_s
            k = thres(j);

            labMap = bwlabel(ucm <= k, 4);
            
            for m = 1:2
               tmp = imdilate(labMap,el);
               labMap(labMap == 0) = tmp(labMap == 0);
            end
            
            [PRI, VOI] = match_segmentations2(labMap, gt_sub);
            grid_PRI(1,j,s,1) = PRI;
            grid_VOI(1,j,s,1) = VOI;
            grid_nLab(1,j,s,1) = numel(unique(labMap(:)));
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));
            grid_COV(1,j,s,1) = COV;
        end % j

        fprintf('UCM: (img %d, sub %d): best_COV = %f\n', i, s, max(max(grid_COV(:,:,s,1))));
    end % s
    
    parsave([eval_dir 'grid_img_' name '.mat'],'grid_PRI','grid_VOI','grid_nLab','grid_COV');
end % i
%matlabpool close

%save(['BSDS_test22_' dataset '.mat']); % **** IMPORTANT

% function parsave(fname,grid_PRI,grid_VOI,grid_nLab,grid_COV)
% save(fname,'grid_PRI','grid_VOI','grid_nLab','grid_COV');
function parsave(varargin)
savefile = varargin{1}; % first input argument
for i=2:nargin
    savevar.(inputname(i)) = varargin{i}; % other input arguments
end
save(savefile,'-struct','savevar')

