function test28_llik_compare_paral(dataset, img_range)
%

fp = fopen('log_test28_compare.txt','w');

img_range = sort(img_range);

fprintf('===== %s set =====\n', dataset);
fprintf('===== %d to %d=====\n', img_range(1), img_range(end));
fprintf(fp,'===== %s set =====\n', dataset);
fprintf(fp,'===== %d to %d=====\n', img_range(1), img_range(end));

%% 
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
eval_dir  = ['./output/grid_eval/' dataset '/'];
gt_eval_dir  = ['./output/grid_gt_eval/' dataset '/'];
poe_eval_dir  = ['./output/grid_poe_eval/' dataset '/'];
poe_dir  = ['./output/poe/' dataset '/'];

all_files = dir(ucm2_dir);
mat       = arrayfun(@(x) ~isempty(strfind(x.name, '.mat')), all_files);
all_files = all_files(logical(mat));

nis = length(all_files);
el = strel('diamond',1);

%if matlabpool('size') == 0 % checking to see if my pool is already open
%    matlabpool open 8
%end

for i = img_range
    tic;
    n_alg = 1;

    % TC
    scals = [5e-4 1e-4 5e-5 1e-5];
    n_r = length(scals);
    ps = [exp(linspace(log(0.8), log(0.89), 30)) exp(linspace(log(0.9), log(0.9999), 30))];
    n_s = length(ps);

    % data
    [~,name] = fileparts(all_files(i).name);
    iid = str2double(name);
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
    gauss_lliks = thisTree.llik;
    
    % poe_lliks learning
    block = [15 15];
    if exist([poe_dir 'poelliks_' num2str(block(1)) 'x' num2str(block(2)) '_img_' num2str(i) '_' name '.mat'], 'file')
        load([poe_dir 'poelliks_' num2str(block(1)) 'x' num2str(block(2)) '_img_' num2str(i) '_' name '.mat'],'poe_lliks');
    else
        diam = floor(block(1)/2);
        cid = ceil(prod(block)/2);
        im1 = rgb2gray(img);
        [nrow,ncol] = size(im1);
        [xs,ttt] = myim2col(im1,block);
        centres = ttt(cid,:);
        xs = double(xs)/255;
        xs = xs';
        
        poe_lliks = zeros(thisTree.numTotalNodes,1);
        for n = 1:thisTree.numTotalNodes
            leaves = thisTree.leafsUnder{n};
            pixs = find(ismember(segMap, leaves) == 1);
            patches = xs(ismember(centres, pixs),:);
            
            if isempty(patches)
                pixs = [pixs-diam;pixs+diam;pixs-(nrow*diam);pixs+(nrow*diam)];
                patches = xs(ismember(centres, pixs),:);
            end
            
            w = poe_learn(patches, block, 0);
            
            llik = poe_logprob(w,patches,block);
            poe_lliks(n) = llik;
            %fprintf('llik(%d) = %f\n', n, llik);
        end % n

        save([poe_dir 'poelliks_' num2str(block(1)) 'x' num2str(block(2)) '_img_' num2str(i) '_' name '.mat'],'poe_lliks');
    end
    thisTree.llik = poe_lliks;

    %% select min & max GT
    %gt_lab_cnt = cellfun(@(x) numel(unique(x.Segmentation(:))), groundTruth);
    %[~,ind_lab_cnt] = sort(gt_lab_cnt);
    %ind_min_max = [ind_lab_cnt(1) ind_lab_cnt(end)];
    n_sub = length(groundTruth);
    
    grid_PRI  = zeros(n_r,n_s,n_sub,n_alg);
    grid_VOI  = zeros(n_r,n_s,n_sub,n_alg);
    grid_nLab = zeros(n_r,n_s,n_sub,n_alg);
    grid_COV  = zeros(n_r,n_s,n_sub,n_alg);

    fprintf('===== img %d_%d with %d subj =====\n', i, iid, n_sub);
    fprintf(fp, '===== img %d_%d with %d subj =====\n', i, iid, n_sub);

    for s = 1:n_sub
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{s}.Segmentation);

        % TC-POE
        for j = 1:n_s
            for r = 1:n_r
                p = ps(j);
                scal = scals(r);

                [~,segLabels] = inference_temp(thisTree, p, scal); % 
                
                [PRI, VOI, labMap] = eval_seg(segMap, segLabels, gt_sub);
                grid_PRI(r,j,s,1) = PRI;
                grid_VOI(r,j,s,1) = VOI;
                grid_nLab(r,j,s,1) = numel(unique(segLabels));
                
                [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
                COV = cntR ./ (sumR + (sumR==0));
                grid_COV(r,j,s,1) = COV;
            end % r
        end % j

        fprintf('TC-POE: (img %d, sub %d): best_COV = %f\n', i, s, max(max(grid_COV(:,:,s,1))));
        fprintf(fp, 'TC-POE: (img %d, sub %d): best_COV = %f\n', i, s, max(max(grid_COV(:,:,s,1))));
    end % s
    
    parsave([poe_eval_dir 'grid_img_' num2str(i) '_' name '.mat'],grid_PRI,grid_VOI,grid_nLab,grid_COV);
    fprintf('img %d takes %f sec.\n', iid, toc);
    fprintf(fp, 'img %d takes %f sec.\n', iid, toc);

    %% comparison
    g_scals = [1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4];
    g_ps = [exp(linspace(log(0.0001), log(0.09), 5)) exp(linspace(log(0.1), log(0.79), 35)) exp(linspace(log(0.8), log(0.89), 30)) exp(linspace(log(0.9), log(0.9999), 30))];

    % Gauss
    temp = load([eval_dir 'grid_img_' num2str(i) '_' name '.mat']);
    grid_COV_gauss = getfield(temp, 'grid_COV');
    grid_COV_gauss = grid_COV_gauss(:,:,:,2);
    grid_PRI_gauss = getfield(temp, 'grid_PRI');
    grid_PRI_gauss = grid_PRI_gauss(:,:,:,2);
    grid_VOI_gauss = getfield(temp, 'grid_VOI');
    grid_VOI_gauss = grid_VOI_gauss(:,:,:,2);
    
    % GT-lik
    temp = load([gt_eval_dir 'grid_img_' num2str(i) '_' name '.mat']);
    grid_COV_gt = getfield(temp, 'grid_COV');
    grid_PRI_gt = getfield(temp, 'grid_PRI');
    grid_VOI_gt = getfield(temp, 'grid_VOI');
    
    for s = 1:n_sub
        fprintf('subj %d: Gauss: \n', s);
        fprintf(fp, 'subj %d: Gauss: \n', s);
        [C,sc,p] = ois(grid_COV_gauss, s, g_scals, g_ps, 'COV');
        fprintf('COV = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'COV = %f at scale = %f, p = %f\n', C, sc, p);
        [C,sc,p] = ois(grid_PRI_gauss, s, g_scals, g_ps, 'PRI');
        fprintf('PRI = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'PRI = %f at scale = %f, p = %f\n', C, sc, p);
        [C,sc,p] = ois(grid_VOI_gauss, s, g_scals, g_ps, 'VOI');
        fprintf('VOI = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'VOI = %f at scale = %f, p = %f\n', C, sc, p);
        
        fprintf('subj %d: POE: \n', s);
        fprintf(fp, 'subj %d: POE: \n', s);
        [C,sc,p] = ois(grid_COV, s, scals, ps, 'COV');
        fprintf('COV = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'COV = %f at scale = %f, p = %f\n', C, sc, p);
        [C,sc,p] = ois(grid_PRI, s, scals, ps, 'PRI');
        fprintf('PRI = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'PRI = %f at scale = %f, p = %f\n', C, sc, p);
        [C,sc,p] = ois(grid_VOI, s, scals, ps, 'VOI');
        fprintf('VOI = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'VOI = %f at scale = %f, p = %f\n', C, sc, p);
        
        fprintf('subj %d: GT: \n', s);
        fprintf(fp, 'subj %d: GT: \n', s);
        [C,sc,p] = ois(grid_COV_gt, s, g_scals, g_ps, 'COV');
        fprintf('COV = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'COV = %f at scale = %f, p = %f\n', C, sc, p);
        [C,sc,p] = ois(grid_PRI_gt, s, g_scals, g_ps, 'PRI');
        fprintf('PRI = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'PRI = %f at scale = %f, p = %f\n', C, sc, p);
        [C,sc,p] = ois(grid_VOI_gt, s, g_scals, g_ps, 'VOI');
        fprintf('VOI = %f at scale = %f, p = %f\n', C, sc, p);
        fprintf(fp, 'VOI = %f at scale = %f, p = %f\n', C, sc, p);
    end

end % i
%matlabpool close

%save(['BSDS_test25_' dataset '.mat']); % **** IMPORTANT
fclose(fp);
%exit

% function parsave(fname,grid_PRI,grid_VOI,grid_nLab,grid_COV)
% save(fname,'grid_PRI','grid_VOI','grid_nLab','grid_COV');
function parsave(varargin)
savefile = varargin{1}; % first input argument
for i=2:nargin
    savevar.(inputname(i)) = varargin{i}; % other input arguments
end
save(savefile,'-struct','savevar')


function [C,sc,p] = ois(grid_res, s, scals, ps, metric)
%
D = squeeze(grid_res(:,:,s,1));
[C,I] = opt(D(:), metric);
[A,B] = ind2sub(size(D),I);
sc = scals(A);
p = ps(B);

function [val, I] = opt(x, metric)
if strcmp(metric,'VOI') == 1
    x(x==0) = +Inf;
    [val, I] = min(x);
else
    [val, I] = max(x);
end

