function test27_sampleMAP_paral(dataset,img_s, img_t)
if nargin < 3
    dataset = 'train';
    img_range = [1 2];
end

img_range = [str2num(img_s):str2num(img_t)];

fprintf('===== %s set =====\n', dataset);
fprintf('===== %d to %d=====\n', img_range(1), img_range(end));

%% 
img_dir  = ['./data/images/' dataset '/'];
ucm2_dir = ['./data/ucm2/' dataset '/'];
gt_dir   = ['./data/groundTruth/' dataset '/'];
tree_dir = ['./output/trees/' dataset '/'];
pt_dir   = ['./output/processed_trees/' dataset '/'];
eval_dir  = ['./output/grid_eval_sampMAP/' dataset '/'];

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
    scals = [1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4];
    n_r = length(scals);
    ps = [exp(linspace(log(0.0001), log(0.09), 5)) exp(linspace(log(0.1), log(0.79), 35)) exp(linspace(log(0.8), log(0.89), 30)) exp(linspace(log(0.9), log(0.9999), 30))];
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

    for s = 1:n_sub
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{s}.Segmentation);

        % TC
        for j = 1:n_s
            if ps(j) < 0.8
                continue
            end
            for r = 1:n_r
                if r == 1 || r == 10
                    continue
                end

                p = ps(j);
                scal = scals(r);

                [aftTree,segLabels] = inference(thisTree, p, scal); % 

                N = 20;
                samples = post_sample(aftTree, N);
                PRI_r_j = zeros(N,1);
                VOI_r_j = zeros(N,1);
                nLab_r_j = zeros(N,1);
                COV_r_j = zeros(N,1);
                for n = 1:N
                    [PRI, VOI, labMap] = eval_seg(segMap, samples(n,:), gt_sub);
                    
                    [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
                    COV = cntR ./ (sumR + (sumR==0));

                    PRI_r_j(n) = PRI;
                    VOI_r_j(n) = VOI;
                    COV_r_j(n) = COV;
                    nLab_r_j(n) = numel(unique(samples(n,:)));
                end

                grid_PRI(r,j,s,1) = mean(PRI_r_j);
                grid_VOI(r,j,s,1) = mean(VOI_r_j);
                grid_nLab(r,j,s,1) = mean(nLab_r_j); 
                grid_COV(r,j,s,1) = mean(COV_r_j);
            end % r
        end % j

        fprintf('TC-sampMAP: (img %d, sub %d): best_COV = %f\n', i, s, max(max(grid_COV(:,:,s,1))));
    end % s
    
    parsave([eval_dir 'grid_img_' num2str(i) '_' name '.mat'],grid_PRI,grid_VOI,grid_nLab,grid_COV);
    fprintf('img %d takes %f sec.\n', iid, toc);
end % i
%matlabpool close

%save(['BSDS_test22_' dataset '.mat']); % **** IMPORTANT
exit

% function parsave(fname,grid_PRI,grid_VOI,grid_nLab,grid_COV)
% save(fname,'grid_PRI','grid_VOI','grid_nLab','grid_COV');
function parsave(varargin)
savefile = varargin{1}; % first input argument
for i=2:nargin
    savevar.(inputname(i)) = varargin{i}; % other input arguments
end
save(savefile,'-struct','savevar')

