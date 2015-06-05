function test27_sampleMAP_paral(dataset,img_s, img_t, metric, schem)
if nargin < 3
    dataset = 'train';
    img_range = [1 2];
end
if nargin < 4
    metric = 'COV';
end
if nargin < 5
    schem = 'ODS';
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
grid_dir  = ['./output/grid_eval/' dataset '/'];
eval_dir  = ['./output/samp_eval_ois/' dataset '/'];

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
    N = 20;

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

    samp_PRI  = zeros(N,n_sub);
    samp_VOI  = zeros(N,n_sub);
    samp_nLab = zeros(N,n_sub);
    samp_COV  = zeros(N,n_sub);

    % grid_metric
    grid_metric = ['grid_' metric];
    fil_nam = [grid_dir 'grid_img_' num2str(i) '_' name '.mat'];
    temp = load(fil_nam, grid_metric);
    grid_res = getfield(temp, grid_metric);

    p_g = 0.971283;
    scal_g = 5e-4;

    % TC
    scals = [1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4];
    ps = [exp(linspace(log(0.0001), log(0.09), 5)) exp(linspace(log(0.1), log(0.79), 35)) exp(linspace(log(0.8), log(0.89), 30))               exp(linspace(log(0.9), log(0.9999), 30))];

    fprintf('===== img %d_%d with %d subj =====\n', i, iid, n_sub);

    for s = 1:n_sub
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{s}.Segmentation);

        % TC-samp
        if strcmp(schem, 'OIS')
            temp = squeeze(grid_res(:,:,s,2));
            [~,I] = opt(temp(:),metric);
            [ri,ji] = ind2sub(size(temp),I);
            p = ps(ji);
            scal = scals(ri);
        else
            p = p_g;
            scal = scal_g;
        end

        [aftTree,segLabels] = inference(thisTree, p, scal); % 

        samples = post_sample(aftTree, N);

        for n = 1:N
            [PRI, VOI, labMap] = eval_seg(segMap, samples(n,:), gt_sub);
            
            [cntR, sumR] = covering_rate_ois(labMap, gt_sub);
            COV = cntR ./ (sumR + (sumR==0));

            samp_PRI(n,s) = PRI;
            samp_VOI(n,s) = VOI;
            samp_COV(n,s) = COV;
            samp_nLab(n,s) = numel(unique(samples(n,:)));
        end

        fprintf('TC-samp: (img %d, sub %d): min_COV = %f ave_COV = %f max_COV = %f\n', i, s, min(samp_COV(:,s)), mean(samp_COV(:,s)), max(samp_COV(:,s)));

    end % s
    
    parsave([eval_dir 'samp_img_' num2str(i) '_' name '.mat'],samp_PRI,samp_VOI,samp_nLab,samp_COV);
    fprintf('img %d takes %f sec.\n', iid, toc);
end % i
%matlabpool close

%save(['BSDS_test22_' dataset '.mat']); % **** IMPORTANT
exit

% function parsave(fname,samp_PRI,samp_VOI,samp_nLab,samp_COV)
% save(fname,'samp_PRI','samp_VOI','samp_nLab','samp_COV');
function parsave(varargin)
savefile = varargin{1}; % first input argument
for i=2:nargin
    savevar.(inputname(i)) = varargin{i}; % other input arguments
end
save(savefile,'-struct','savevar')


function [val, I] = opt(x, metric)
if strcmp(metric,'VOI') == 1
    x(x==0) = +Inf;
    [val, I] = min(x);
else
    [val, I] = max(x);
end

