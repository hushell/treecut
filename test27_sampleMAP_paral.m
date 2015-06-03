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
eval_dir  = ['./output/samp_eval_sampMAP/' dataset '/'];

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

    p_g = 0.971283;
    scal_g = 5e-4;

    fprintf('===== img %d_%d with %d subj =====\n', i, iid, n_sub);

    for s = 1:n_sub
        gt_sub = cell(1,1);
        gt_sub{1}.Segmentation = double(groundTruth{s}.Segmentation);

        % TC-samp
        p = p_g;
        scal = scal_g;

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

